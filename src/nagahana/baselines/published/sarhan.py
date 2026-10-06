"""Extra trees on the NetFlow standard feature sets (Sarhan, Layeghy and Portmann, Mobile Networks and
Applications 27(1):357-370, 2022, "Towards a Standard Feature Set for Network Intrusion Detection System
Datasets", DOI 10.1007/s11036-021-01843-0; values from arXiv:2101.11315v2).

What the paper states (baselines-notes.md, B2)
    model         extra-trees classifier, binary detection
    features      the NetFlow v2 set (or the dataset's original features), without flow identifiers
                  (addresses, ports, time stamps); for the UNSW-NB15 variants also without TTL features
    scaling       min-max
    protocol      random 70 %/30 % train/test split, five splits, mean reported
    metrics       accuracy, AUC, F1, DR (detection rate = recall), FAR (false-alarm rate = FPR)

What the paper leaves open (AS-536)
    n_estimators = 50 and scikit-learn defaults for every other hyperparameter; L7_PROTO and the flag
    fields are min-max scaled like every other column (the paper scales all features); non-finite rows
    are dropped from training.

The printed AUC
    In all four rows of Table 8 the printed AUC equals (DR + 1 - FAR) / 2: 0.9545 = (0.9125 + 0.9965) / 2,
    0.9845 = (0.9707 + 0.9984) / 2 (0.98455 printed to four places), 0.9684 = (0.9475 + 0.9893) / 2 and
    0.9829 = (0.9689 + 0.9969) / 2. That is the area under the ROC curve of the hard decisions, i.e. the
    balanced accuracy, so the hook compares it with balanced accuracy (metric "auc_of_decisions").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from nagahana.baselines.published.base import BaselineConfig, BaselineSpec, InputSchema, Reference, ReportedResult
from nagahana.baselines.published.features import cicflowmeter, netflow, unsw_nb15
from nagahana.baselines.published.protocols import RandomHoldout, SplitProtocol
from nagahana.baselines.published.tabular import TabularBaseline, TabularConfig

FeatureSet = Literal["netflow-v2", "netflow-v1", "cicflowmeter", "unsw-nb15-full"]


@dataclass
class SarhanConfig(TabularConfig):
    """Extra trees with min-max scaling on a NetFlow (or original) feature set.

    Attributes
    ----------
    feature_set:
        "netflow-v2" (43 features, NF-*-v2 files), "netflow-v1" (12 features), "cicflowmeter" (original
        CSE-CIC-IDS2018 features) or "unsw-nb15-full" (original UNSW-NB15 features).
    drop_ttl:
        Remove the TTL-based features (the paper does so for the UNSW-NB15 variants).
    """

    learner: str = "extra_trees"
    params: dict[str, Any] = field(default_factory=lambda: {"n_estimators": 50})
    scaler: Literal["none", "minmax", "standard", "l2"] = "minmax"
    label_column: str = "Label"
    family_column: str = "Attack"
    feature_set: FeatureSet = "netflow-v2"
    drop_ttl: bool = False


REFERENCE = Reference(
    key="sarhan2022standard",
    authors="Sarhan, Layeghy, Portmann",
    title="Towards a Standard Feature Set for Network Intrusion Detection System Datasets",
    venue="Mobile Networks and Applications 27(1):357-370",
    year=2022,
    doi="10.1007/s11036-021-01843-0",
    arxiv="2101.11315",
    note="values taken from arXiv:2101.11315v2",
)

_PROTOCOL = "random 70/30 split, mean of five splits; binary; identifiers removed"


def _row(dataset: str, acc: str, auc: str, f1: str, dr: str, far: str, variant: dict[str, Any]) -> ReportedResult:
    return ReportedResult(
        dataset=dataset, protocol=_PROTOCOL, task="binary", model="extra_trees",
        values={"accuracy": acc, "auc_of_decisions": auc, "f1": f1, "dr": dr, "far": far},
        location="Tab. 8, p. 8 (arXiv v2)", variant=variant,
        note="printed AUC equals (DR + 1 - FAR) / 2, compared with balanced accuracy",
    )


SPEC = BaselineSpec(
    name="sarhan-extra-trees",
    title="Extra trees on the NetFlow standard feature set",
    reference=REFERENCE,
    family="flow-classifier",
    input_schema=InputSchema(
        description="One row per flow record: the NetFlow v2 (or v1) columns of the NF-* datasets, or the original "
                    "CICFlowMeter / UNSW-NB15 columns, with the binary `Label` (0 benign, 1 attack) and the class name "
                    "in `Attack` where the files carry it.",
        label="Label",
        optional=("Attack", "time", "dataset", "network", "split"),
    ),
    outputs=("detection",),
    datasets=("unsw-nb15", "nf-unsw-nb15-v2", "cse-cic-ids2018", "nf-cse-cic-ids2018-v2"),
    reported=(
        _row("unsw-nb15", "99.25%", "0.9545", "0.92", "91.25%", "0.35%", {"feature_set": "unsw-nb15-full", "drop_ttl": True}),
        _row("nf-unsw-nb15-v2", "99.73%", "0.9845", "0.97", "97.07%", "0.16%", {"feature_set": "netflow-v2", "drop_ttl": True}),
        _row("cse-cic-ids2018", "98.31%", "0.9684", "0.94", "94.75%", "1.07%", {"feature_set": "cicflowmeter", "drop_ttl": False}),
        _row("nf-cse-cic-ids2018-v2", "99.35%", "0.9829", "0.97", "96.89%", "0.31%", {"feature_set": "netflow-v2", "drop_ttl": False}),
    ),
    third_party="sarhan-netflow-v2",
    assumptions=("AS-530", "AS-532", "AS-533", "AS-536"),
)


class SarhanExtraTrees(TabularBaseline):
    """Sarhan et al. 2022: extra trees, min-max scaling, identifiers (and TTL for UNSW-NB15) removed."""

    spec: ClassVar[BaselineSpec] = SPEC
    config_type: ClassVar[type[BaselineConfig]] = SarhanConfig
    config: SarhanConfig

    def default_features(self, columns: Sequence[str]) -> tuple[str, ...]:
        cfg = self.config
        have = set(columns)
        if cfg.feature_set == "netflow-v2":
            feats = netflow.standard_features(2, drop_identifiers=True, drop_ttl=cfg.drop_ttl)
        elif cfg.feature_set == "netflow-v1":
            feats = netflow.standard_features(1, drop_identifiers=True, drop_ttl=False)
        elif cfg.feature_set == "cicflowmeter":
            feats = cicflowmeter.feature_columns(list(columns), drop=cicflowmeter.CIC_IDENTIFIERS)
        else:
            feats = unsw_nb15.full_features(drop_identifiers=True, drop_ttl=cfg.drop_ttl)
        if cfg.feature_set in ("netflow-v2", "netflow-v1", "unsw-nb15-full"):
            netflow.check_columns(have, feats)
        return feats

    @classmethod
    def paper_protocol(cls) -> SplitProtocol:
        """Random 70/30 split, five repeats (baselines-notes.md, B2)."""
        return RandomHoldout(test_fraction=0.3, repeats=5)
