"""Hidden-Markov-model forecasters of multistep attacks.

Three published systems share one engine (core.py):

    Holgado, Villagra and Vazquez, IEEE TDSC 17(1):134-147, 2020, "Real-time multistep attack prediction
    based on hidden Markov models" (DOI 10.1109/TDSC.2017.2751478; citation to verify): hidden states are
    the stages of a multistep attack, observations are IDS alerts; Baum-Welch training; online
    filtering gives the probability that the attack reaches its final stage.

    Chadza, Kyriakopoulos and Lambotharan, IEEE Access 8:134480-134497, 2020, "Learning to Learn
    Sequential Network Attacks Using Hidden Markov Models" (DOI 10.1109/ACCESS.2020.3011293): transfer
    learning (TL) of an HMM from a labelled source capture to an unlabelled target, against conventional
    learning on the target only (CML); training by Baum-Welch (BW) or DE with uniform, random or
    count-based starts; next-state (NS) and next-observation (NO) predictions scored as top-1, 2, 3
    accuracy; a sliding window of 150 observations; 30/70 and 70/30 splits with two-fold swapping.

    Ghafir, Kyriakopoulos, Lambotharan, Aparicio-Navarro, AsSadhan, Binsalleeh and Diab, IEEE Access
    7:99508-99520, 2019, "Hidden Markov Models and Alert Correlations for the Prediction of Advanced
    Persistent Threats" (DOI 10.1109/ACCESS.2019.2930200): six APT stages as states, eleven alert types as
    observations; Viterbi decoding of the stage sequence; next-stage prediction (one-stage = top-1,
    two-stage = top-2) after 2, 3 and 4 alerts; 50/50 train / test.

Input: one row per observation, grouped into sequences (`sequence_column`) and ordered by time; the
observation is an alert code (`observation_column`) or, where the data carry no alerts, a feature vector
turned into a symbol by a k-means codebook (symbols.py, AS-547). An optional stage column gives the true
stage of each observation (training labels where the method is supervised, and evaluation labels).

Outputs per observation (unit "window", one row per row of the input)
    stage        filtered P(stage_t | o_{..t}) (a sliding window of `context_window` observations, or the
                 whole history), in the paper's stage names or, with `attack_stage_map`, in the ATT&CK
                 vocabulary of models/vocab.py (stages.py)
    forecast     P_inf(k), k = 1 ... K: the probability of entering an infiltration stage within k steps
                 (first passage, core.first_passage), with hazards; targets from the stage labels
    detection    P(s_t in an infiltration stage | o_{..t})
    component    next_stage_probs / next_stage_label (NS), next_observation_probs / next_observation_label
                 (NO), viterbi_stage, position (index in the sequence), state_belief, decoded_prefix_match

State identity (AS-546): with supervised or count-based estimation the hidden states are the stages by
construction. After unsupervised training, each state is assigned to the stage with which its posterior
mass co-occurs most on the labelled training steps (the many-to-one mapping used to evaluate unsupervised
HMMs); without labels, state i keeps the name of stage i.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd

from nagahana.baselines.published.base import (
    BaselineConfig,
    BaselineSpec,
    InputSchema,
    PredictionParts,
    PublishedBaseline,
    Reference,
    ReportedResult,
    derive_seed,
)
from nagahana.baselines.published.frames import build_meta, epoch_seconds, normalise_token
from nagahana.baselines.published.hmm.core import (
    HMMParams,
    baum_welch,
    filter_stream,
    first_passage,
    initial_params,
    left_to_right_mask,
    posteriors,
    supervised_estimate,
    viterbi,
)
from nagahana.baselines.published.hmm.evolution import differential_evolution
from nagahana.baselines.published.hmm.symbols import KMeansCodebook
from nagahana.baselines.published.protocols import Split, SplitProtocol
from nagahana.baselines.published.stages import attack_stages, infiltration_stage_names, name_projection
from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import DetectionPredictions, ForecastPredictions, StagePredictions

Training = Literal["baum_welch", "supervised", "differential_evolution"]
Init = Literal["uniform", "random", "count"]


@dataclass
class HMMForecasterConfig(BaselineConfig):
    """Settings shared by the HMM forecasters.

    Attributes
    ----------
    state_names:
        Hidden states, one per attack stage, in stage order.
    attack_stage_map:
        Stage name -> ATT&CK tactic name of models/vocab.py; empty keeps the paper's names.
    infiltration_states:
        Stage names whose entry is the forecast event; empty derives them from the map (tactics of the
        infiltration set, AS-18).
    sequence_column, time_column, observation_column, stage_column:
        Input columns (sequence and time optional: one sequence in file order without them).
    feature_columns, n_symbols, codebook_init:
        Feature vectors symbolised by a k-means codebook of n_symbols codes (n_init restarts); without
        feature columns the observation column holds codes and n_symbols = 0 takes max code + 1.
    training, init, restarts:
        Estimation ("baum_welch", "supervised", "differential_evolution"), its start and, for random
        starts of Baum-Welch, the number of restarts (the best objective is kept).
    topology, max_jump:
        "ergodic" or "left_to_right" transitions (a stage stays or advances, by at most max_jump stages
        when max_jump > 0).
    pseudo_count, max_iter, tol, symmetry_breaking:
        Dirichlet pseudo-counts of every estimate, EM limits, and the relative perturbation that breaks
        the symmetry of a uniform start (a uniform ergodic start is a fixed point of EM).
    de_population, de_generations, de_mutation, de_crossover, de_spread:
        Differential evolution (evolution.py).
    context_window:
        Observations the filter keeps (0: the whole history).
    horizon, step_seconds:
        Forecast steps K and the duration attributed to one step in the forecast record.
    threshold:
        Operating threshold on P(infiltration stage now).
    map_states_with_labels:
        Map unsupervised states to stages with the training labels when present.
    """

    state_names: tuple[str, ...] = ()
    attack_stage_map: dict[str, str] = field(default_factory=dict)
    infiltration_states: tuple[str, ...] = ()
    sequence_column: str = "sequence"
    time_column: str = "time"
    observation_column: str = "observation"
    stage_column: str = "stage"
    feature_columns: tuple[str, ...] = ()
    n_symbols: int = 0
    codebook_init: int = 4
    training: Training = "baum_welch"
    init: Init = "uniform"
    restarts: int = 1
    topology: Literal["ergodic", "left_to_right"] = "ergodic"
    max_jump: int = 0
    pseudo_count: float = 1e-3
    max_iter: int = 200
    tol: float = 1e-6
    symmetry_breaking: float = 1e-3
    de_population: int = 20
    de_generations: int = 50
    de_mutation: float = 0.5
    de_crossover: float = 0.9
    de_spread: float = 0.5
    context_window: int = 0
    horizon: int = 12
    step_seconds: float = 60.0
    threshold: float = 0.5
    map_states_with_labels: bool = True

    def validate(self) -> None:
        super().validate()
        if len(self.state_names) < 2 or len(set(self.state_names)) != len(self.state_names):
            raise ValueError("state_names must name at least two distinct stages")
        unknown = [s for s in self.attack_stage_map if s not in self.state_names]
        if unknown:
            raise ValueError(f"attack_stage_map names unknown stages {unknown}")
        if self.attack_stage_map and set(self.attack_stage_map) != set(self.state_names):
            raise ValueError("attack_stage_map must map every stage or none")
        if any(s not in self.state_names for s in self.infiltration_states):
            raise ValueError("infiltration_states must be stage names")
        if not self.infiltration_states and not self.attack_stage_map:
            raise ValueError("give infiltration_states or an attack_stage_map to derive them from")
        if self.feature_columns and self.n_symbols < 1:
            raise ValueError("symbolising feature columns needs n_symbols >= 1")
        if self.n_symbols < 0 or self.restarts < 1 or self.max_iter < 1 or self.horizon < 1 or self.context_window < 0:
            raise ValueError("n_symbols >= 0, restarts >= 1, max_iter >= 1, horizon >= 1, context_window >= 0")
        if self.pseudo_count < 0 or self.symmetry_breaking < 0 or self.step_seconds <= 0 or self.max_jump < 0:
            raise ValueError("pseudo_count, symmetry_breaking, max_jump must be >= 0 and step_seconds > 0")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")

    @property
    def n_states(self) -> int:
        return len(self.state_names)


def _groups(frame: pd.DataFrame, sequence_column: str, time_column: str) -> list[np.ndarray]:
    """Row positions of each sequence in time order; sequences in order of their first observation."""
    n = len(frame)
    t = epoch_seconds(frame[time_column]) if time_column in frame.columns else np.arange(n, dtype=np.float64)
    keys = frame[sequence_column].astype(str).to_numpy() if sequence_column in frame.columns else np.zeros(n, dtype=object)
    codes, _ = pd.factorize(pd.Series(keys), sort=False)
    order = np.lexsort((np.arange(n), t, codes))                               # by sequence, then time, then file order
    sorted_codes = codes[order]
    bounds = np.r_[0, np.nonzero(np.diff(sorted_codes))[0] + 1, n]
    groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    first = [float(t[g].min()) for g in groups]
    return [groups[i] for i in np.argsort(first, kind="stable")]


def majority_mapping(params: HMMParams, obs: list[np.ndarray], stages: list[np.ndarray]) -> np.ndarray:
    """[N, N] many-to-one assignment of hidden states to stages by posterior co-occurrence.

    co[i, s] = sum over labelled steps t with stage s of P(s_t = i | o); state i is assigned to argmax_s
    co[i, s]; a state without posterior mass on labelled steps keeps stage i.
    """
    n = params.n_states
    co = np.zeros((n, n))
    for o, s in zip(obs, stages, strict=True):
        known = s >= 0
        if not known.any():
            continue
        gamma, _, _ = posteriors(params, o)
        np.add.at(co.T, s[known], gamma[known])                                 # co[i, stage] += gamma_t(i)
    out = np.eye(n)
    for i in range(n):
        if co[i].sum() > 0:
            out[i] = 0.0
            out[i, int(np.argmax(co[i]))] = 1.0
    return out


def forecast_targets(stage: np.ndarray, target_stage: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """(event_step, observed_steps) for every position of one stage sequence.

    For position t, step k looks at stage[t + k]. Steps are observed while the future stages are known
    (the first unknown stage or the end of the sequence censors the rest); event_step is the first
    observed step whose stage is a target stage (0 when none), and observed_steps is that step, or the
    number of observed steps when no event occurs.
    """
    length = stage.size
    fut = np.full((length, horizon), -1, dtype=np.int64)
    for k in range(1, horizon + 1):
        if k < length:
            fut[: length - k, k - 1] = stage[k:]
    known_prefix = np.cumprod(fut >= 0, axis=1).astype(bool)                     # contiguous known steps
    hit = known_prefix & target_stage[np.clip(fut, 0, None)]
    any_hit = hit.any(axis=1)
    event = np.where(any_hit, np.argmax(hit, axis=1) + 1, 0).astype(np.int64)
    observed = np.where(any_hit, event, known_prefix.sum(axis=1)).astype(np.int64)
    return event, observed


class HMMForecaster(PublishedBaseline):
    """The shared engine of the HMM forecasters (see the module docstring)."""

    config_type: ClassVar[type[BaselineConfig]] = HMMForecasterConfig
    config: HMMForecasterConfig

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config)
        self.params: HMMParams | None = None
        self.codebook: KMeansCodebook | None = None
        self.n_symbols_ = 0
        self.state_stage_: np.ndarray = np.eye(self.config.n_states)            # [N, S] 0/1 mapping

    def required_columns(self) -> tuple[str, ...]:
        cfg = self.config
        return tuple(cfg.feature_columns) if cfg.feature_columns else (cfg.observation_column,)

    def _mask(self) -> np.ndarray | None:
        cfg = self.config
        if cfg.topology == "ergodic":
            return None
        return left_to_right_mask(cfg.n_states, max_jump=cfg.max_jump or None)

    def _infiltration_mask(self) -> np.ndarray:
        """[S] bool over stages: the forecast's target stages."""
        cfg = self.config
        if cfg.infiltration_states:
            return np.asarray([s in cfg.infiltration_states for s in cfg.state_names])
        infil = infiltration_stage_names()
        return np.asarray([cfg.attack_stage_map[s] in infil for s in cfg.state_names])

    def _stage_index(self, frame: pd.DataFrame) -> np.ndarray:
        """Stage code per row (index into state_names), -1 where unknown or absent."""
        cfg = self.config
        if cfg.stage_column not in frame.columns:
            return np.full(len(frame), -1, dtype=np.int64)
        names = {normalise_token(s): i for i, s in enumerate(cfg.state_names)}
        out = np.full(len(frame), -1, dtype=np.int64)
        for i, v in enumerate(frame[cfg.stage_column].tolist()):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if isinstance(v, int | np.integer) and not isinstance(v, bool):
                out[i] = int(v) if 0 <= int(v) < cfg.n_states else -1
            else:
                out[i] = names.get(normalise_token(v), -1)
        return out

    def _fit_symbols(self, frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        cfg = self.config
        if cfg.feature_columns:
            x = frame.loc[:, list(cfg.feature_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
            self.codebook = KMeansCodebook(cfg.n_symbols, n_init=cfg.codebook_init).fit(x, rng)
            self.n_symbols_ = cfg.n_symbols
            return self.codebook.transform(x)
        codes = self._observation_codes(frame, fitting=True)
        if cfg.n_symbols and int(codes.max()) >= cfg.n_symbols:
            raise InvariantViolation(f"{self.spec.name}: observation code {int(codes.max())} exceeds n_symbols = {cfg.n_symbols}")
        self.n_symbols_ = cfg.n_symbols or int(codes.max()) + 1
        return codes

    def _observation_codes(self, frame: pd.DataFrame, *, fitting: bool = False) -> np.ndarray:
        cfg = self.config
        if self.codebook is not None and cfg.feature_columns:
            x = frame.loc[:, list(cfg.feature_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
            return self.codebook.transform(x)
        raw = pd.to_numeric(frame[cfg.observation_column], errors="coerce").to_numpy(dtype=np.float64)
        codes = np.where(np.isfinite(raw), raw, -1).astype(np.int64)
        if np.any(np.isfinite(raw) & (raw != np.round(raw))) or codes.min(initial=0) < -1:
            raise InvariantViolation("observation codes must be non-negative integers (-1 or empty for missing)")
        if not fitting and self.n_symbols_ and codes.max(initial=-1) >= self.n_symbols_:
            # A symbol never seen in training is treated as a missing observation (no emission evidence).
            codes = np.where(codes >= self.n_symbols_, -1, codes)
        if fitting and codes.max(initial=-1) < 0:
            raise InvariantViolation("training data hold no observation")
        return codes

    def _start(self, obs: list[np.ndarray], stages: list[np.ndarray], rng: np.random.Generator) -> HMMParams:
        """Starting parameters of the configured kind."""
        cfg = self.config
        mask = self._mask()
        if cfg.init == "count":
            if not any((s >= 0).any() for s in stages):
                raise InvariantViolation(f"{self.spec.name}: a count-based start needs labelled training steps")
            return supervised_estimate(obs, stages, cfg.n_states, self.n_symbols_, pseudo=max(cfg.pseudo_count, 1e-12),
                                       transition_mask=mask)
        p = initial_params(cfg.n_states, self.n_symbols_, cfg.init, rng, transition_mask=mask)
        if cfg.init == "uniform" and cfg.symmetry_breaking > 0:
            # Multiplicative perturbation of the emissions: a fully uniform ergodic start is an EM fixed point.
            b = p.B * np.exp(cfg.symmetry_breaking * rng.standard_normal(p.B.shape))
            p = HMMParams(p.pi, p.A, b / b.sum(axis=1, keepdims=True))
        return p

    def _train(self, obs: list[np.ndarray], stages: list[np.ndarray], rng: np.random.Generator,
               start: HMMParams | None = None) -> HMMParams:
        cfg = self.config
        mask = self._mask()
        if cfg.training == "supervised":
            if not any((s >= 0).any() for s in stages):
                raise InvariantViolation(f"{self.spec.name}: supervised estimation needs labelled training steps")
            p = supervised_estimate(obs, stages, cfg.n_states, self.n_symbols_, pseudo=cfg.pseudo_count, transition_mask=mask)
            self.fit_report["estimation"] = "counting"
            return p
        first = start if start is not None else self._start(obs, stages, rng)
        if cfg.training == "differential_evolution":
            res = differential_evolution(obs, first, rng, population=cfg.de_population, generations=cfg.de_generations,
                                         mutation=cfg.de_mutation, crossover=cfg.de_crossover, spread=cfg.de_spread,
                                         transition_mask=mask)
            self.fit_report["de_best_fitness"] = res.best_fitness
            return res.params
        best: tuple[float, HMMParams, list[float]] | None = None
        for r in range(cfg.restarts):
            init = first if r == 0 else initial_params(cfg.n_states, self.n_symbols_, "random",
                                                       np.random.default_rng(derive_seed(cfg.seed, "restart", r)), transition_mask=mask)
            p, hist = baum_welch(obs, init, max_iter=cfg.max_iter, tol=cfg.tol, pseudo_pi=cfg.pseudo_count,
                                 pseudo_a=cfg.pseudo_count, pseudo_b=cfg.pseudo_count, transition_mask=mask)
            if best is None or hist.objective[-1] > best[0]:
                best = (hist.objective[-1], p, hist.loglik)
        assert best is not None
        self.fit_report["em_loglik"] = best[2]
        return best[1]

    def _state_mapping(self, params: HMMParams, obs: list[np.ndarray], stages: list[np.ndarray]) -> np.ndarray:
        """[N, S] 0/1 assignment of hidden states to stages (module docstring)."""
        cfg = self.config
        if cfg.training == "supervised" or cfg.init == "count" or not cfg.map_states_with_labels:
            return np.eye(cfg.n_states)
        return majority_mapping(params, obs, stages)

    def _fit(self, data: pd.DataFrame, validation: pd.DataFrame | None, rng: np.random.Generator) -> None:
        cfg = self.config
        symbols = self._fit_symbols(data, rng)
        stage = self._stage_index(data)
        groups = _groups(data, cfg.sequence_column, cfg.time_column)
        obs = [symbols[g] for g in groups]
        stages = [stage[g] for g in groups]
        self.params = self._train(obs, stages, rng)
        self.state_stage_ = self._state_mapping(self.params, obs, stages)
        self.fit_report.update({"sequences": len(groups), "observations": int(sum(len(g) for g in groups)),
                                "symbols": self.n_symbols_, "labelled_steps": int((stage >= 0).sum())})

    def _predict(self, data: pd.DataFrame) -> PredictionParts:
        assert self.params is not None
        cfg = self.config
        p = self.params
        n, n_states, k_max = len(data), cfg.n_states, cfg.horizon
        symbols = self._observation_codes(data)
        stage = self._stage_index(data)
        groups = _groups(data, cfg.sequence_column, cfg.time_column)
        belief = np.zeros((n, n_states))
        viterbi_stage = np.full(n, -1, dtype=np.int64)
        position = np.zeros(n, dtype=np.int64)
        seq_index = np.zeros(n, dtype=np.int64)
        next_stage_label = np.full(n, -1, dtype=np.int64)
        next_obs_label = np.full(n, -1, dtype=np.int64)
        event_step = np.zeros(n, dtype=np.int64)
        observed_steps = np.zeros(n, dtype=np.int64)
        target_stage = self._infiltration_mask()                                # [S]
        decoded_match = np.full((len(groups), 5), -1, dtype=np.int64)
        state_to_stage = np.argmax(self.state_stage_, axis=1)                   # hard assignment per state
        for gi, g in enumerate(groups):
            o = symbols[g]
            belief[g] = filter_stream(p, o, window=cfg.context_window or None)
            path, _ = viterbi(p, o)
            viterbi_stage[g] = state_to_stage[path]
            position[g] = np.arange(g.size)
            seq_index[g] = gi
            s = stage[g]
            next_stage_label[g[:-1]] = s[1:]
            next_obs_label[g[:-1]] = o[1:]
            # Forecast targets: first future step in an infiltration stage, censored at the first unknown.
            event_step[g], observed_steps[g] = forecast_targets(s, target_stage, k_max)
            # Decoded-sequence accuracy after the first m observations (Ghafir et al., Tab. 3).
            for m in range(1, min(5, g.size) + 1):
                if (s[:m] >= 0).all():
                    prefix, _ = viterbi(p, o[:m])
                    decoded_match[gi, m - 1] = int(np.array_equal(state_to_stage[prefix], s[:m]))
        stage_probs = belief @ self.state_stage_                                # [n, S]
        next_stage_probs = (belief @ p.A) @ self.state_stage_
        next_obs_probs = belief @ p.A @ p.B
        target_states = (self.state_stage_ @ target_stage.astype(np.float64)) > 0
        p_inf = first_passage(p, belief, target_states, k_max)
        prev = np.concatenate([np.zeros((n, 1)), p_inf[:, :-1]], axis=1)
        hazard = np.clip(np.divide(p_inf - prev, 1.0 - prev, out=np.ones_like(p_inf), where=(1.0 - prev) > 1e-12), 0.0, 1.0)
        meta = build_meta(data, time_column=cfg.time_column if cfg.time_column in data.columns else None,
                          family=np.asarray(data["family"].astype(str)) if "family" in data.columns else np.full(n, "unknown"),
                          extra={"sequence": seq_index, "position": position})
        names = tuple(cfg.state_names)
        if cfg.attack_stage_map:
            proj = name_projection(names, cfg.attack_stage_map)                  # [S, 15]
            out_probs, out_names = stage_probs @ proj, attack_stages()
            out_label = np.where(stage >= 0, np.argmax(proj[np.clip(stage, 0, None)], axis=1), -1)
        else:
            out_probs, out_names, out_label = stage_probs, names, stage
        det_label = np.where(stage >= 0, target_stage[np.clip(stage, 0, None)].astype(np.int64), -1)
        return PredictionParts(
            detection=DetectionPredictions(score=np.clip(stage_probs @ target_stage.astype(np.float64), 0.0, 1.0), label=det_label,
                                           unit="window", meta=meta, threshold=cfg.threshold),
            stage=StagePredictions(probs=out_probs, label=out_label, stage_names=out_names, meta=meta),
            forecast=ForecastPredictions(p_inf=p_inf, window_seconds=cfg.step_seconds, event_step=event_step,
                                         observed_steps=observed_steps, meta=meta, hazard=hazard),
            component={"state_belief": belief, "paper_stage_probs": stage_probs, "next_stage_probs": next_stage_probs,
                       "next_stage_label": next_stage_label, "next_observation_probs": next_obs_probs,
                       "next_observation_label": next_obs_label, "viterbi_stage": viterbi_stage, "position": position,
                       "decoded_prefix_match": decoded_match},
        )

    def _export_state(self, directory: Path) -> dict[str, Any]:
        assert self.params is not None
        return {"params": self.params.state(), "n_symbols": self.n_symbols_, "state_stage": self.state_stage_.tolist(),
                "codebook": None if self.codebook is None else self.codebook.state()}

    def _import_state(self, directory: Path, state: Mapping[str, Any]) -> None:
        self.params = HMMParams.from_state(state["params"])
        self.n_symbols_ = int(state["n_symbols"])
        self.state_stage_ = np.asarray(state["state_stage"], dtype=np.float64)
        self.codebook = None if state["codebook"] is None else KMeansCodebook.from_state(state["codebook"])


#: The five phases of the DARPA 2000 LLDOS 1.0 scenario and their tactics (AS-549).
LLDOS_PHASES: tuple[str, ...] = ("ip_sweep", "sadmind_probe", "break_in", "ddos_install", "ddos_launch")
LLDOS_TACTICS: dict[str, str] = {"ip_sweep": "reconnaissance", "sadmind_probe": "reconnaissance", "break_in": "initial_access",
                                 "ddos_install": "execution", "ddos_launch": "impact"}
#: The six APT stages of Ghafir et al. and their tactics (AS-551).
APT_STAGES: tuple[str, ...] = ("intelligence_gathering", "point_of_entry", "command_and_control", "lateral_movement",
                               "asset_discovery", "data_exfiltration")
APT_TACTICS: dict[str, str] = {"intelligence_gathering": "reconnaissance", "point_of_entry": "initial_access",
                               "command_and_control": "command_and_control", "lateral_movement": "lateral_movement",
                               "asset_discovery": "discovery", "data_exfiltration": "exfiltration"}


@dataclass
class HolgadoConfig(HMMForecasterConfig):
    """Holgado et al. 2020: left-to-right stages of a multistep attack, Baum-Welch from counted starts."""

    state_names: tuple[str, ...] = LLDOS_PHASES
    attack_stage_map: dict[str, str] = field(default_factory=lambda: dict(LLDOS_TACTICS))
    infiltration_states: tuple[str, ...] = ("ddos_launch",)
    training: Training = "baum_welch"
    init: Init = "count"
    topology: Literal["ergodic", "left_to_right"] = "left_to_right"


HOLGADO_REFERENCE = Reference(
    key="holgado2020realtime",
    authors="Holgado, Villagra, Vazquez",
    title="Real-time multistep attack prediction based on hidden Markov models",
    venue="IEEE Transactions on Dependable and Secure Computing 17(1):134-147",
    year=2020,
    doi="10.1109/TDSC.2017.2751478",
    note="citation to verify; not transcribed in baselines-notes.md, so no reported values are tabulated",
)

_HMM_SCHEMA = InputSchema(
    description="One row per observation (an IDS alert code, or a feature vector symbolised by a k-means codebook), "
                "with a `sequence` id (attack campaign, host or stream), event `time` and, where known, the attack "
                "`stage`.",
    optional=("sequence", "time", "stage", "family", "domain", "dataset", "network", "split"),
)

HOLGADO_SPEC = BaselineSpec(
    name="holgado-hmm",
    title="Real-time multistep attack prediction with a hidden Markov model",
    reference=HOLGADO_REFERENCE,
    family="stage-forecaster",
    input_schema=_HMM_SCHEMA,
    outputs=("detection", "stage", "forecast"),
    datasets=("darpa-2000",),
    reported=(),
    third_party="holgado-hmm",
    assumptions=("AS-546", "AS-547", "AS-548", "AS-549"),
)


class HolgadoHMM(HMMForecaster):
    """Holgado et al. 2020 (see the module docstring)."""

    spec: ClassVar[BaselineSpec] = HOLGADO_SPEC
    config_type: ClassVar[type[BaselineConfig]] = HolgadoConfig
    config: HolgadoConfig

    @classmethod
    def paper_protocol(cls) -> SplitProtocol:
        """Random 50/50 split of the observation sequences (AS-549)."""
        return SequenceHoldout(test_fraction=0.5)


@dataclass
class GhafirConfig(HMMForecasterConfig):
    """Ghafir et al. 2019: six APT stages, supervised estimation on the labelled training half."""

    state_names: tuple[str, ...] = APT_STAGES
    attack_stage_map: dict[str, str] = field(default_factory=lambda: dict(APT_TACTICS))
    training: Training = "supervised"
    init: Init = "count"
    n_symbols: int = 11


GHAFIR_REFERENCE = Reference(
    key="bl-ghafir2019apt",
    authors="Ghafir, Kyriakopoulos, Lambotharan, Aparicio-Navarro, AsSadhan, Binsalleeh, Diab",
    title="Hidden Markov Models and Alert Correlations for the Prediction of Advanced Persistent Threats",
    venue="IEEE Access 7:99508-99520",
    year=2019,
    doi="10.1109/ACCESS.2019.2930200",
)


def _ghafir_rows() -> tuple[ReportedResult, ...]:
    p = "synthetic alerts (3,700 APT + 2,300 uncorrelated), 50/50 train/test"
    rows = []
    for after, top1, top2 in (("2", "43.60%", "66.50%"), ("3", "72.77%", "92.70%"), ("4", "93.31%", "100%")):
        rows.append(ReportedResult(dataset="ghafir-synthetic-6000", protocol=p, task="next-stage", model="hmm",
                                   values={f"top1_next_stage_after_{after}": top1, f"top2_next_stage_after_{after}": top2},
                                   location="Tab. 4, p. 99518"))
    rows.append(ReportedResult(dataset="ghafir-synthetic-6000", protocol=p, task="stage-decoding", model="hmm",
                               values={"decoded_sequence_accuracy_after_2": "91.80%", "decoded_sequence_accuracy_after_3": "100%",
                                       "decoded_sequence_accuracy_after_4": "100%", "decoded_sequence_accuracy_after_5": "100%"},
                               location="Tab. 3, p. 99517"))
    return tuple(rows)


GHAFIR_SPEC = BaselineSpec(
    name="ghafir-hmm",
    title="HMM prediction of the next APT stage from correlated alerts",
    reference=GHAFIR_REFERENCE,
    family="stage-forecaster",
    input_schema=_HMM_SCHEMA,
    outputs=("detection", "stage", "forecast"),
    datasets=("ghafir-synthetic-6000",),
    reported=_ghafir_rows(),
    third_party="ghafir-hmm",
    assumptions=("AS-546", "AS-547", "AS-548", "AS-551"),
)


class GhafirHMM(HMMForecaster):
    """Ghafir et al. 2019 (see the module docstring)."""

    spec: ClassVar[BaselineSpec] = GHAFIR_SPEC
    config_type: ClassVar[type[BaselineConfig]] = GhafirConfig
    config: GhafirConfig

    @classmethod
    def paper_protocol(cls) -> SplitProtocol:
        """Two equal halves for training and testing, by whole sequences (baselines-notes.md, C3)."""
        return SequenceHoldout(test_fraction=0.5)


@dataclass
class ChadzaConfig(HMMForecasterConfig):
    """Chadza et al. 2020: transfer from a labelled source capture to an unlabelled target.

    Attributes
    ----------
    transfer:
        True for TL (source model counted on the labelled source, refined on the target), False for CML
        (target only).
    domain_column, source_domain, target_domain:
        Column and values separating the two captures.
    """

    state_names: tuple[str, ...] = LLDOS_PHASES
    attack_stage_map: dict[str, str] = field(default_factory=lambda: dict(LLDOS_TACTICS))
    training: Training = "baum_welch"
    init: Init = "uniform"
    context_window: int = 150
    transfer: bool = True
    domain_column: str = "domain"
    source_domain: str = "source"
    target_domain: str = "target"


CHADZA_REFERENCE = Reference(
    key="bl-chadza2020hmm",
    authors="Chadza, Kyriakopoulos, Lambotharan",
    title="Learning to Learn Sequential Network Attacks Using Hidden Markov Models",
    venue="IEEE Access 8:134480-134497",
    year=2020,
    doi="10.1109/ACCESS.2020.3011293",
)

CHADZA_SPEC = BaselineSpec(
    name="chadza-hmm",
    title="Transfer learning of attack-stage HMMs (Baum-Welch or differential evolution)",
    reference=CHADZA_REFERENCE,
    family="stage-forecaster",
    input_schema=_HMM_SCHEMA,
    outputs=("detection", "stage", "forecast"),
    datasets=("darpa-2000", "cse-cic-ids2018"),
    reported=(
        ReportedResult(dataset="darpa-2000 inside->dmz", protocol="TL; target split 30/70 or 70/30, window 150, two-fold swap",
                       task="next-stage", model="tl-de", values={"top3_next_stage": "96.3%"}, location="Sec. VII-A, p. 134491",
                       variant={"training": "differential_evolution", "transfer": True},
                       note="the authors write 'about 96.3%' (uniform and count-based DE)"),
        ReportedResult(dataset="darpa-2000 inside->dmz", protocol="TL; target split 30/70 or 70/30, window 150, two-fold swap",
                       task="next-observation", model="tl-bw-uniform", values={"top3_next_observation": "79.9%"},
                       location="Sec. VII-A, p. 134492", variant={"training": "baum_welch", "init": "uniform", "transfer": True}),
        ReportedResult(dataset="cse-cic-ids2018 cic1->cic2", protocol="TL; target split, window 150, two-fold swap",
                       task="next-observation", model="tl-de-uniform", values={"top3_next_observation": "81.5%"},
                       location="Sec. VII-B, p. 134495", variant={"training": "differential_evolution", "init": "uniform", "transfer": True}),
        ReportedResult(dataset="cse-cic-ids2018 cic2", protocol="CML on the target only", task="next-observation",
                       model="cml-de-uniform", values={"top3_next_observation": "20.6%"}, location="Sec. VII-B, p. 134495",
                       variant={"training": "differential_evolution", "init": "uniform", "transfer": False}),
    ),
    third_party="chadza-hmm",
    assumptions=("AS-546", "AS-547", "AS-548", "AS-550"),
)


class ChadzaHMM(HMMForecaster):
    """Chadza et al. 2020: TL (source counts -> target refinement) or CML (target only)."""

    spec: ClassVar[BaselineSpec] = CHADZA_SPEC
    config_type: ClassVar[type[BaselineConfig]] = ChadzaConfig
    config: ChadzaConfig

    def _transfer_start(self, source: HMMParams, src_obs: list[np.ndarray], tgt_obs: list[np.ndarray],
                        rng: np.random.Generator) -> HMMParams:
        """The source model, with emissions of target-only symbols initialised by `init` (AS-550).

        For symbols k never seen in the source, B0[i, k] = rho q_k, where rho is the share of target
        observations with such symbols and q is uniform, random (Dirichlet) or the target's count
        distribution over them; the source emissions are scaled by 1 - rho, so every row sums to 1.
        """
        cfg = self.config
        m = self.n_symbols_
        src_seen = np.zeros(m, dtype=bool)
        for o in src_obs:
            src_seen[o[o >= 0]] = True
        tgt_all = np.concatenate([o[o >= 0] for o in tgt_obs]) if tgt_obs else np.zeros(0, dtype=np.int64)
        tgt_counts = np.bincount(tgt_all, minlength=m).astype(np.float64)
        new = ~src_seen
        if not new.any() or tgt_counts.sum() == 0:
            return source
        rho = float(tgt_counts[new].sum() / tgt_counts.sum())
        if cfg.init == "uniform":
            q = np.where(new, 1.0, 0.0)
        elif cfg.init == "random":
            q = np.zeros(m)
            q[new] = rng.dirichlet(np.ones(int(new.sum())))
        else:
            q = np.where(new, tgt_counts + cfg.pseudo_count, 0.0)
        q = q / q.sum()
        b = np.where(new[None, :], 0.0, source.B)
        b = b / b.sum(axis=1, keepdims=True) * (1.0 - rho) + rho * q[None, :]
        return HMMParams(source.pi, source.A, b)

    def _fit(self, data: pd.DataFrame, validation: pd.DataFrame | None, rng: np.random.Generator) -> None:
        cfg = self.config
        if cfg.domain_column not in data.columns:
            raise InvariantViolation(f"{self.spec.name}: training frame needs a {cfg.domain_column!r} column")
        dom = data[cfg.domain_column].astype(str).to_numpy()
        symbols = self._fit_symbols(data, rng)
        stage = self._stage_index(data)
        groups = _groups(data, cfg.sequence_column, cfg.time_column)
        src_g = [g[dom[g] == cfg.source_domain] for g in groups]
        tgt_g = [g[dom[g] == cfg.target_domain] for g in groups]
        src_g, tgt_g = [g for g in src_g if g.size], [g for g in tgt_g if g.size]
        if not tgt_g:
            raise InvariantViolation(f"{self.spec.name}: no target-domain rows to train on")
        tgt_obs = [symbols[g] for g in tgt_g]
        no_labels = [np.full(g.size, -1, dtype=np.int64) for g in tgt_g]           # the target is unlabelled
        if cfg.transfer:
            if not src_g:
                raise InvariantViolation(f"{self.spec.name}: transfer needs labelled source-domain rows")
            src_obs, src_stage = [symbols[g] for g in src_g], [stage[g] for g in src_g]
            source = supervised_estimate(src_obs, src_stage, cfg.n_states, self.n_symbols_, pseudo=max(cfg.pseudo_count, 1e-12),
                                         transition_mask=self._mask())
            start = self._transfer_start(source, src_obs, tgt_obs, rng)
            if cfg.training == "supervised":
                self.params = start
            else:
                self.params = self._train(tgt_obs, no_labels, rng, start=start)
            self.state_stage_ = np.eye(cfg.n_states)
        else:
            if cfg.init == "count":
                # CML count-based start: emissions from the target's symbol counts (with symmetry breaking).
                counts = np.bincount(np.concatenate([o[o >= 0] for o in tgt_obs]), minlength=self.n_symbols_) + cfg.pseudo_count
                base = initial_params(cfg.n_states, self.n_symbols_, "uniform", rng, transition_mask=self._mask())
                b = np.tile(counts / counts.sum(), (cfg.n_states, 1)) * np.exp(
                    max(cfg.symmetry_breaking, 1e-3) * rng.standard_normal((cfg.n_states, self.n_symbols_)))
                start = HMMParams(base.pi, base.A, b / b.sum(axis=1, keepdims=True))
            else:
                start = self._start(tgt_obs, no_labels, rng)
            self.params = self._train(tgt_obs, no_labels, rng, start=start)
            # The target's training labels, when present, only name the learned states (evaluation mapping).
            tgt_stage = [stage[g] for g in tgt_g]
            self.state_stage_ = majority_mapping(self.params, tgt_obs, tgt_stage) if cfg.map_states_with_labels else np.eye(cfg.n_states)
        self.fit_report.update({"transfer": cfg.transfer, "source_sequences": len(src_g), "target_sequences": len(tgt_g),
                                "symbols": self.n_symbols_})

    @classmethod
    def paper_protocol(cls) -> SplitProtocol:
        """Target split 30/70 by position in the stream, with the two parts swapped (two folds)."""
        return SequenceFractionSwap(train_fraction=0.3)


@dataclass(frozen=True)
class SequenceHoldout(SplitProtocol):
    """Random hold-out of whole sequences (column `sequence`); rows without it form one sequence."""

    test_fraction: float = 0.5
    sequence_column: str = "sequence"

    def splits(self, frame: pd.DataFrame, seed: int) -> Iterator[Split]:
        ids = frame[self.sequence_column].astype(str).to_numpy() if self.sequence_column in frame.columns else np.zeros(len(frame), dtype=object)
        names = np.asarray(sorted(set(ids.tolist())))
        rng = np.random.default_rng(derive_seed(seed, "sequence-holdout"))
        names = names[rng.permutation(names.size)]
        k = max(1, int(round(self.test_fraction * names.size)))
        test = np.isin(ids, names[:k])
        yield Split("sequence-holdout", np.nonzero(~test)[0], np.nonzero(test)[0])

    def describe(self) -> str:
        return f"random {1 - self.test_fraction:.0%}/{self.test_fraction:.0%} hold-out of whole sequences"


@dataclass(frozen=True)
class SequenceFractionSwap(SplitProtocol):
    """Each sequence's first `train_fraction` of observations (in time order) vs the rest, then swapped.

    Chadza et al. split the target capture 30/70 or 70/30 into training and evaluation parts and swap
    them in a second fold; source-domain rows (column `domain` equal to "source") always stay in training.
    """

    train_fraction: float = 0.3
    sequence_column: str = "sequence"
    time_column: str = "time"
    domain_column: str = "domain"
    source_domain: str = "source"

    def splits(self, frame: pd.DataFrame, seed: int) -> Iterator[Split]:
        part = np.zeros(len(frame), dtype=np.int64)                             # 0: first part, 1: second part
        is_source = (frame[self.domain_column].astype(str).to_numpy() == self.source_domain
                     if self.domain_column in frame.columns else np.zeros(len(frame), dtype=bool))
        for g in _groups(frame, self.sequence_column, self.time_column):
            g = g[~is_source[g]]
            cut = int(round(self.train_fraction * g.size))
            part[g[cut:]] = 1
        src = np.nonzero(is_source)[0]
        first, second = np.nonzero((part == 0) & ~is_source)[0], np.nonzero((part == 1) & ~is_source)[0]
        yield Split("first-part-trains", np.sort(np.concatenate([src, first])), second)
        yield Split("second-part-trains", np.sort(np.concatenate([src, second])), first)

    def describe(self) -> str:
        f = self.train_fraction
        return f"target split {f:.0%}/{1 - f:.0%} in time order, swapped (two folds)"


__all__ = [
    "APT_STAGES", "APT_TACTICS", "LLDOS_PHASES", "LLDOS_TACTICS", "ChadzaConfig", "ChadzaHMM", "GhafirConfig", "GhafirHMM",
    "HMMForecaster", "HMMForecasterConfig", "HolgadoConfig", "HolgadoHMM", "SequenceFractionSwap", "SequenceHoldout",
]
