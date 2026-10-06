"""PPO and GNN-PPO defense baselines in CyberWheel (the comparison baselines of CyberWorld, AS-559).

Proximal policy optimisation (Schulman, Wolski, Dhariwal, Radford and Klimov, arXiv:1707.06347) in the
single-file form of CleanRL (Huang et al., JMLR 23(274), 2022; "The 37 Implementation Details of Proximal
Policy Optimization", ICLR Blog Track 2022), with N parallel environments and rollouts of T steps:

    advantages   GAE: delta_t = r_t + gamma V(s_{t+1}) (1 - d_{t+1}) - V(s_t),
                 A_t = delta_t + gamma lambda (1 - d_{t+1}) A_{t+1};  returns R_t = A_t + V(s_t)
    policy loss  E[max(-A r_t(theta), -A clip(r_t(theta), 1 - eps, 1 + eps))],  r_t = pi(a|s) / pi_old(a|s),
                 advantages normalised per minibatch
    value loss   0.5 E[max((V - R)^2, (V_old + clip(V - V_old, -eps, eps) - R)^2)]
    total        policy loss - c_ent H[pi] + c_v value loss; Adam with linearly annealed learning rate,
                 global gradient-norm clipping
CleanRL defaults: T = 128, gamma 0.99, lambda 0.95, 4 minibatches, 4 epochs, eps 0.2, c_ent 0.01, c_v 0.5,
learning rate 2.5e-4, clip norm 0.5; orthogonal initialisation (gain sqrt 2, policy head 0.01, value head 1).

Policies
    mlp   actor and critic: two tanh layers of 64 units on the flat observation (CyberWorld's PPO
          baseline: 128 parallel environments, 3.2M environment steps)
    gnn   a graph-attention encoder over the host slots (node features: the five host fields and the
          broadcast telemetry; topology: the adjacency); critic on mean and max pooling of the nodes; actor
          with one deploy and one remove logit per slot and a no-op logit from the pooled embedding when
          the decoy targets are the slots, a pooled head over all actions otherwise (GNN-PPO: 52,500 steps)
Truncated episodes are treated as terminated, as in CleanRL. Settings the CyberWorld paper does not
state are those of CleanRL (citation to verify).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import torch
from torch import nn

from nagahana.baselines.published.base import BaselineConfig, Reference
from nagahana.baselines.published.cyberwheel.agents import AgentSpec, DefenseAgent
from nagahana.baselines.published.cyberwheel.env import (
    DefenseObservation,
    EnvFactory,
    VectorEnv,
    batch_graphs,
    batch_vectors,
)
from nagahana.baselines.published.cyberworld.networks import GATEncoder
from nagahana.baselines.published.neural import load_module, resolve_device, save_module


def _ortho(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class MLPActorCritic(nn.Module):
    """CleanRL's discrete actor-critic: two tanh layers each for actor and critic."""

    def __init__(self, d_in: int, n_actions: int, hidden: int) -> None:
        super().__init__()
        g = math.sqrt(2.0)
        self.critic = nn.Sequential(_ortho(nn.Linear(d_in, hidden), g), nn.Tanh(), _ortho(nn.Linear(hidden, hidden), g),
                                    nn.Tanh(), _ortho(nn.Linear(hidden, 1), 1.0))
        self.actor = nn.Sequential(_ortho(nn.Linear(d_in, hidden), g), nn.Tanh(), _ortho(nn.Linear(hidden, hidden), g),
                                   nn.Tanh(), _ortho(nn.Linear(hidden, n_actions), 0.01))

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """(logits [B, A], value [B])."""
        x = obs["vector"]
        return self.actor(x), self.critic(x)[:, 0]


class GNNActorCritic(nn.Module):
    """Graph-attention actor-critic over host slots (module docstring)."""

    def __init__(self, n_fields: int, telemetry_dim: int, n_slots: int, n_actions: int, hidden: int, layers: int,
                 heads: int) -> None:
        super().__init__()
        g = math.sqrt(2.0)
        self.encoder = GATEncoder(n_fields + telemetry_dim, hidden, layers=layers, heads=heads)
        self.per_slot = n_actions == 1 + 2 * n_slots
        self.critic = nn.Sequential(_ortho(nn.Linear(2 * hidden, hidden), g), nn.Tanh(), _ortho(nn.Linear(hidden, 1), 1.0))
        if self.per_slot:
            self.node_head = nn.Sequential(_ortho(nn.Linear(hidden, hidden), g), nn.Tanh(), _ortho(nn.Linear(hidden, 2), 0.01))
            self.noop_head = nn.Sequential(_ortho(nn.Linear(2 * hidden, hidden), g), nn.Tanh(), _ortho(nn.Linear(hidden, 1), 0.01))
        else:
            self.global_head = nn.Sequential(_ortho(nn.Linear(2 * hidden, hidden), g), nn.Tanh(),
                                             _ortho(nn.Linear(hidden, n_actions), 0.01))

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hosts, adj, tel = obs["hosts"], obs["adjacency"], obs["telemetry"]
        b, v, _ = hosts.shape
        x = torch.cat([hosts, tel[:, None, :].expand(b, v, tel.shape[-1])], dim=-1)
        mask = torch.ones(b, v, dtype=torch.bool, device=hosts.device)
        h = self.encoder(x, adj, mask)                                              # [B, V, hidden]
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)            # [B, 2 hidden]
        value = self.critic(pooled)[:, 0]
        if self.per_slot:
            node = self.node_head(h)                                                # [B, V, 2]: deploy, remove
            logits = torch.cat([self.noop_head(pooled), node[..., 0], node[..., 1]], dim=-1)
        else:
            logits = self.global_head(pooled)
        return logits, value


@dataclass
class PPOConfig(BaselineConfig):
    """PPO settings (module docstring); `policy` selects the MLP or the GNN actor-critic."""

    policy: Literal["mlp", "gnn"] = "mlp"
    num_envs: int = 128
    total_steps: int = 3_200_000
    num_steps: int = 128
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 64
    gnn_layers: int = 2
    gnn_heads: int = 4

    def validate(self) -> None:
        super().validate()
        if min(self.num_envs, self.num_steps, self.num_minibatches, self.update_epochs, self.hidden) < 1:
            raise ValueError("num_envs, num_steps, num_minibatches, update_epochs and hidden must be >= 1")
        if self.total_steps < self.num_envs * self.num_steps:
            raise ValueError("total_steps must cover at least one rollout (num_envs * num_steps)")
        if (self.num_envs * self.num_steps) % self.num_minibatches:
            raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
        if not (0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1 and self.clip_coef > 0 and self.learning_rate > 0):
            raise ValueError("gamma in (0, 1], gae_lambda in [0, 1], clip_coef > 0 and learning_rate > 0")


@dataclass
class GNNPPOConfig(PPOConfig):
    """GNN-PPO: the graph policy with the 52,500-step budget."""

    policy: Literal["mlp", "gnn"] = "gnn"
    num_envs: int = 8
    total_steps: int = 52_500
    num_steps: int = 128


CYBERWORLD_REFERENCE = Reference(
    key="masukawa2026cyberworld",
    authors="Masukawa, Yun, Hassan, Oh, Jeong, Imani",
    title="CyberWorld",
    venue="arXiv preprint",
    year=2026,
    arxiv="2609.31893",
    note="citation to verify (title, venue and settings were not read in this build); CC BY 4.0",
)

PPO_SPEC = AgentSpec(name="cyberwheel-ppo", title="PPO defense agent (MLP policy)", reference=CYBERWORLD_REFERENCE,
                     environment="cyberwheel", budget_steps=3_200_000, third_party="cyberwheel", assumptions=("AS-558", "AS-559"))
GNN_PPO_SPEC = AgentSpec(name="cyberwheel-gnn-ppo", title="GNN-PPO defense agent (graph-attention policy)",
                         reference=CYBERWORLD_REFERENCE, environment="cyberwheel", budget_steps=52_500, third_party="cyberwheel",
                         assumptions=("AS-558", "AS-559"))


def _to_tensors(observations: list[DefenseObservation], policy: str, device: torch.device) -> dict[str, torch.Tensor]:
    if policy == "mlp":
        return {"vector": torch.as_tensor(batch_vectors(observations), device=device)}
    hosts, adj, tel = batch_graphs(observations)
    return {"hosts": torch.as_tensor(hosts, device=device), "adjacency": torch.as_tensor(adj, device=device),
            "telemetry": torch.as_tensor(tel, device=device)}


class PPOAgent(DefenseAgent):
    """CleanRL PPO with the MLP or the GNN policy (module docstring)."""

    spec: ClassVar[AgentSpec] = PPO_SPEC
    config_type: ClassVar[type[BaselineConfig]] = PPOConfig
    config: PPOConfig

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config)
        self.net: nn.Module | None = None
        self.shape: dict[str, int] = {}

    def _build(self) -> nn.Module:
        cfg, s = self.config, self.shape
        if cfg.policy == "mlp":
            return MLPActorCritic(s["vector_dim"], s["n_actions"], cfg.hidden)
        return GNNActorCritic(s["n_fields"], s["telemetry_dim"], s["n_slots"], s["n_actions"], cfg.hidden, cfg.gnn_layers, cfg.gnn_heads)

    def _train(self, factory: EnvFactory, rng: np.random.Generator) -> None:
        cfg = self.config
        device = resolve_device(cfg.device)
        envs = VectorEnv(factory, cfg.num_envs, cfg.seed)
        obs = envs.reset()
        first = obs[0]
        self.shape = {"vector_dim": int(first.vector().size), "n_actions": envs.action_space.n, "n_slots": first.n_slots,
                      "n_fields": int(first.hosts.shape[1]), "telemetry_dim": int(first.telemetry.size)}
        net = self._build().to(device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, eps=1e-5)
        n, t_len = cfg.num_envs, cfg.num_steps
        batch = n * t_len
        mb = batch // cfg.num_minibatches
        updates = cfg.total_steps // batch
        done = np.zeros(n, dtype=bool)
        history: list[dict[str, float]] = []
        for update in range(updates):
            if cfg.anneal_lr:
                for gpar in opt.param_groups:
                    gpar["lr"] = (1.0 - update / updates) * cfg.learning_rate
            store_obs: list[list[DefenseObservation]] = []
            actions = np.zeros((t_len, n), dtype=np.int64)
            logprobs = np.zeros((t_len, n))
            rewards = np.zeros((t_len, n))
            dones = np.zeros((t_len, n))
            values = np.zeros((t_len, n))
            for step in range(t_len):
                store_obs.append(obs)
                dones[step] = done
                with torch.no_grad():
                    logits, value = net(_to_tensors(obs, cfg.policy, device))
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                actions[step] = a.cpu().numpy()
                logprobs[step] = dist.log_prob(a).cpu().numpy()
                values[step] = value.cpu().numpy()
                obs, rew, term, trunc, _ = envs.step(actions[step].tolist())
                rewards[step] = rew
                done = term | trunc
            with torch.no_grad():
                _, next_value_t = net(_to_tensors(obs, cfg.policy, device))
            next_value = next_value_t.cpu().numpy()
            adv = np.zeros((t_len, n))
            last = np.zeros(n)
            for t in reversed(range(t_len)):
                non_terminal = 1.0 - (done.astype(np.float64) if t == t_len - 1 else dones[t + 1])
                next_v = next_value if t == t_len - 1 else values[t + 1]
                delta = rewards[t] + cfg.gamma * next_v * non_terminal - values[t]
                last = delta + cfg.gamma * cfg.gae_lambda * non_terminal * last
                adv[t] = last
            returns = adv + values
            flat_obs = [o for row in store_obs for o in row]                        # order: step-major
            b_actions = torch.as_tensor(actions.reshape(-1), device=device)
            b_logprobs = torch.as_tensor(logprobs.reshape(-1), dtype=torch.float32, device=device)
            b_adv = torch.as_tensor(adv.reshape(-1), dtype=torch.float32, device=device)
            b_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device)
            b_values = torch.as_tensor(values.reshape(-1), dtype=torch.float32, device=device)
            stats: dict[str, float] = {}
            for _epoch in range(cfg.update_epochs):
                perm = torch.as_tensor(rng.permutation(batch), device=device)
                for start in range(0, batch, mb):
                    idx = perm[start:start + mb]
                    logits, new_value = net(_to_tensors([flat_obs[i] for i in idx.tolist()], cfg.policy, device))
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logprob = dist.log_prob(b_actions[idx])
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(new_logprob - b_logprobs[idx])
                    a_mb = b_adv[idx]
                    if cfg.norm_adv:
                        a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                    pg_loss = torch.max(-a_mb * ratio, -a_mb * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
                    if cfg.clip_vloss:
                        v_clip = b_values[idx] + torch.clamp(new_value - b_values[idx], -cfg.clip_coef, cfg.clip_coef)
                        v_loss = 0.5 * torch.max((new_value - b_returns[idx]) ** 2, (v_clip - b_returns[idx]) ** 2).mean()
                    else:
                        v_loss = 0.5 * ((new_value - b_returns[idx]) ** 2).mean()
                    loss = pg_loss - cfg.ent_coef * entropy + cfg.vf_coef * v_loss
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                    opt.step()
                    stats = {"policy_loss": float(pg_loss.detach()), "value_loss": float(v_loss.detach()),
                             "entropy": float(entropy.detach()), "approx_kl": float(((ratio - 1) - torch.log(ratio)).mean().detach())}
            ep = envs.stats.summary()
            history.append({"update": float(update), "env_steps": float((update + 1) * batch), **stats,
                            "episode_return_mean": ep["return_mean"]})
        self.net = net
        self.train_report.update({"updates": updates, "env_steps": updates * batch, "history": history,
                                  "episodes": envs.stats.summary()})

    @torch.no_grad()
    def _act(self, observation: DefenseObservation, greedy: bool) -> int:
        assert self.net is not None
        device = resolve_device(self.config.device)
        logits, _ = self.net(_to_tensors([observation], self.config.policy, device))
        if greedy:
            return int(torch.argmax(logits, dim=-1)[0])
        return int(torch.distributions.Categorical(logits=logits).sample()[0])

    def _export_state(self, directory: Path) -> dict[str, Any]:
        assert self.net is not None
        return {"weights": save_module(self.net, directory / "policy.pt"), "shape": self.shape}

    def _import_state(self, directory: Path, state: Mapping[str, Any]) -> None:
        self.shape = {k: int(v) for k, v in state["shape"].items()}
        self.net = load_module(self._build(), directory / str(state["weights"]), resolve_device(self.config.device))
        self.net.eval()


class GNNPPOAgent(PPOAgent):
    """GNN-PPO: PPO with the graph-attention policy and the 52,500-step budget."""

    spec: ClassVar[AgentSpec] = GNN_PPO_SPEC
    config_type: ClassVar[type[BaselineConfig]] = GNNPPOConfig
    config: GNNPPOConfig
