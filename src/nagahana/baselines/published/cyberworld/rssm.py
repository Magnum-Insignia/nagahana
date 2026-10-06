"""Recurrent state-space model of CyberWorld (DreamerV3 form; Hafner et al., arXiv:2301.04104).

State s_t = (h_t, z_t): h_t deterministic (1,024 units in CyberWorld), z_t G categorical variables with C
classes each (32 x 32), represented one-hot.

    h_t   = GRU(h_{t-1}, MLP([z_{t-1} ; a_{t-1}]))                          sequence model
    p(z_t | h_t)      = Cat(mix(softmax(prior(h_t))))                       dynamics predictor (prior)
    q(z_t | h_t, e_t) = Cat(mix(softmax(post([h_t ; e_t]))))                representation (posterior)
    mix(p) = (1 - u) p + u / C,  u = 0.01                                   "unimix"
Samples are one-hot with straight-through gradients: z = onehot(sample) + p - stopgrad(p).

KL terms (summed over the G variables, then floored at `free_nats`):
    L_dyn = max(free, KL(stopgrad(q) || p)),   L_rep = max(free, KL(q || stopgrad(p)))
with weights beta_dyn = 0.5 and beta_rep = 0.1 in the world-model loss (world.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from nagahana.baselines.published.cyberworld.networks import MLP, NormGRUCell


@dataclass
class RSSMState:
    """h [..., deter], logits [..., G, C], z [..., G * C] (one-hot sample)."""

    h: torch.Tensor
    logits: torch.Tensor
    z: torch.Tensor

    def features(self) -> torch.Tensor:
        """[h ; z], the input of every head."""
        return torch.cat([self.h, self.z], dim=-1)

    def detach(self) -> RSSMState:
        return RSSMState(self.h.detach(), self.logits.detach(), self.z.detach())


def stack_states(states: list[RSSMState], dim: int) -> RSSMState:
    """Stack a list of states along a new axis."""
    return RSSMState(torch.stack([s.h for s in states], dim), torch.stack([s.logits for s in states], dim),
                     torch.stack([s.z for s in states], dim))


class RSSM(nn.Module):
    """The model of the module docstring; `action_dim` 0 gives an action-free (forecasting) RSSM."""

    def __init__(self, *, deter: int, groups: int, classes: int, embed: int, action_dim: int, units: int,
                 unimix: float = 0.01) -> None:
        super().__init__()
        self.deter, self.groups, self.classes, self.unimix = deter, groups, classes, unimix
        self.action_dim = action_dim
        stoch = groups * classes
        self.img_in = MLP(stoch + action_dim, units, 1)
        self.gru = NormGRUCell(units, deter)
        self.prior_net = MLP(deter, units, 1, stoch)
        self.post_net = MLP(deter + embed, units, 1, stoch)
        self.h0 = nn.Parameter(torch.zeros(deter))

    @property
    def feature_dim(self) -> int:
        return self.deter + self.groups * self.classes

    def initial(self, batch: int, device: torch.device) -> RSSMState:
        """The learned initial deterministic state (tanh of a parameter) and a zero latent."""
        h = torch.tanh(self.h0).expand(batch, -1)
        logits = torch.zeros(batch, self.groups, self.classes, device=device)
        return RSSMState(h, logits, torch.zeros(batch, self.groups * self.classes, device=device))

    def _probs(self, logits: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.unimix) * F.softmax(logits, dim=-1) + self.unimix / self.classes

    def sample(self, logits: torch.Tensor, *, mode: bool = False) -> torch.Tensor:
        """Straight-through one-hot sample (or the mode) of logits [..., G, C], flattened to [..., G * C]."""
        probs = self._probs(logits)
        if mode:
            idx = probs.argmax(dim=-1)
        else:
            idx = torch.distributions.Categorical(probs=probs).sample()
        onehot = F.one_hot(idx, self.classes).to(probs.dtype)
        z = onehot + probs - probs.detach()
        return z.flatten(-2)

    def img_step(self, prev: RSSMState, action: torch.Tensor | None, *, mode: bool = False) -> RSSMState:
        """Prior step: h_t from (h_{t-1}, z_{t-1}, a_{t-1}), then z_t ~ p(z_t | h_t)."""
        x = prev.z if self.action_dim == 0 or action is None else torch.cat([prev.z, action], dim=-1)
        h = self.gru(self.img_in(x), prev.h)
        logits = self.prior_net(h).view(*h.shape[:-1], self.groups, self.classes)
        return RSSMState(h, logits, self.sample(logits, mode=mode))

    def obs_step(self, prev: RSSMState, action: torch.Tensor | None, embed: torch.Tensor) -> tuple[RSSMState, RSSMState]:
        """(posterior, prior) of step t given the embedding e_t."""
        prior = self.img_step(prev, action)
        logits = self.post_net(torch.cat([prior.h, embed], dim=-1)).view(*prior.h.shape[:-1], self.groups, self.classes)
        return RSSMState(prior.h, logits, self.sample(logits)), prior

    def observe(self, embeds: torch.Tensor, actions: torch.Tensor | None, first: torch.Tensor,
                start: RSSMState | None = None) -> tuple[RSSMState, RSSMState]:
        """Filter a batch of sequences: embeds [B, T, E], actions [B, T, A] (a_{t-1} at index t), first [B, T]
        bool (episode starts reset the state). Returns (posteriors, priors) stacked along T.
        """
        b, t_len, _ = embeds.shape
        state = start if start is not None else self.initial(b, embeds.device)
        posts: list[RSSMState] = []
        priors: list[RSSMState] = []
        init = self.initial(b, embeds.device)
        for t in range(t_len):
            reset = first[:, t, None].to(embeds.dtype)
            state = RSSMState(reset * init.h + (1.0 - reset) * state.h,
                              reset[..., None] * init.logits + (1.0 - reset[..., None]) * state.logits,
                              reset * init.z + (1.0 - reset) * state.z)
            act = None if actions is None else actions[:, t] * (1.0 - reset)
            post, prior = self.obs_step(state, act, embeds[:, t])
            posts.append(post)
            priors.append(prior)
            state = post
        return stack_states(posts, 1), stack_states(priors, 1)

    def kl(self, post_logits: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
        """KL(q || p) summed over the G categorical variables, [...]."""
        q = self._probs(post_logits)
        p = self._probs(prior_logits)
        return (q * (torch.log(q) - torch.log(p))).sum(dim=(-2, -1))

    def kl_losses(self, post: RSSMState, prior: RSSMState, free_nats: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(L_dyn, L_rep, raw KL), each [...] (module docstring)."""
        dyn = self.kl(post.logits.detach(), prior.logits)
        rep = self.kl(post.logits, prior.logits.detach())
        raw = self.kl(post.logits.detach(), prior.logits.detach())
        return torch.clamp(dyn, min=free_nats), torch.clamp(rep, min=free_nats), raw
