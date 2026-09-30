"""The training pipeline: six stages, as data (D-22; [I-01], [A-24]).

Why data and not code: each stage's inputs, outputs, trainable and frozen modules, and split usage are
decisions. Keeping them in one declarative table makes the plan reviewable (`python -m nagahana
stages`). It also lets DVC (dvc.yaml), the runners and the tests share one source.

The six stages (from the owner's image [I-01]; ai-mod-arch has the same content in five steps, with
analysis and preparation merged):

1. **analysis**: "we collect raw public/open source datasets, we take them and conduct our analysis
   framework to understand the data, its problems/issues, and all --> document it" [A-24], [Q-02].
2. **preparation**: "cleaning, splitting, extraction of attacks and certain things along with
   normalization for pretraining, data augmentation using the generator" [A-24]. Splits per D-23.
3. **pretrain-perception**: self-supervised pretraining of CVG-AE, Decoder and TSTCT. "teaching it the
   intuition to mentally model the network topology, and its dynamics".
4. **pretrain-taaft**: self-supervised pretraining of TAAFT "with frozen pretrained AE, Decoder & TSTCT"
   [I-01]. "the intuition of adversaries, conflict, cooperation, coordination …".
5. **full-training**: the whole architecture, including the Forecaster and Advisor policy/value agents.
   The Verifier is trained on human feedback (D-07).
6. **zero-shot**: "zero-shot validation dataset which consists of both novel & known attacks" (real
   data only, D-23), with human-supervised calibration for site adaptability (D-13 held).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StagePlan:
    """One stage of the pipeline.

    Attributes
    ----------
    id, name: stable key and title.
    trains: modules whose weights change in this stage.
    frozen: modules loaded from earlier stages and kept fixed.
    splits: data splits read.
    self_supervised: the objective uses no attack labels.
    human_feedback: the stage uses analyst feedback (Verifier), always human-gated (D-21).
    produces: artefacts for DVC/MLflow.
    waiting_on: decisions or specifications that block running the stage.
    """

    id: int
    key: str
    name: str
    trains: tuple[str, ...]
    frozen: tuple[str, ...]
    splits: tuple[str, ...]
    self_supervised: bool
    human_feedback: bool
    produces: tuple[str, ...]
    waiting_on: tuple[str, ...]


STAGES: tuple[StagePlan, ...] = (
    StagePlan(1, "analysis", "Deep data analysis", (), (), ("raw",), False, False,
              ("analysis report", "adapter column maps", "field catalogue extensions"), ("datasets on the workstation",)),
    StagePlan(2, "preparation", "Preparation (with the Generator)", ("generator",), (), ("raw",), True, False,
              ("prepared dataset", "split manifest", "generated variants"), ("D-14", "D-16", "P-23")),
    StagePlan(3, "pretrain-perception", "Self-supervised pretraining: CVG-AE, Decoder, TSTCT",
              ("cvgae", "decoder", "tstct"), (), ("train", "val"), True, False,
              ("perception checkpoint",), ("D-11a", "D-04", "D-15")),
    StagePlan(4, "pretrain-taaft", "Self-supervised pretraining: TAAFT", ("taaft",), ("cvgae", "decoder", "tstct"),
              ("train", "val"), True, False, ("TAAFT checkpoint",), ("stage-4 objectives", "D-11b")),
    StagePlan(5, "full-training", "Full training incl. Forecaster and Advisor agents; Verifier on human feedback",
              ("taaft", "forecaster_heads", "advisor_heads", "verifier"), (), ("train", "val"), False, True,
              ("full checkpoint", "calibration state"), ("D-12", "D-03a", "D-03b", "D-03c", "D-11c")),
    StagePlan(6, "zero-shot", "Zero-shot validation with calibration (novel and known, real data only)",
              ("site_adapter",), ("cvgae", "decoder", "tstct", "taaft", "forecaster_heads", "advisor_heads"),
              ("zero_shot_known", "zero_shot_novel"), False, True, ("evaluation report", "site calibration"),
              ("D-13", "D-16")),
)


def get(key: str | int) -> StagePlan:
    """A stage by key ("pretrain-taaft") or number (4)."""
    for s in STAGES:
        if key in (s.id, s.key):
            return s
    raise KeyError(f"No stage {key!r}; stages: {[s.key for s in STAGES]}")
