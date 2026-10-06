"""Generic tabular reproduction: one row = one scored unit, a feature set, a preprocessing rule, a learner.

The classical-learner studies (Sarhan et al. 2022, Cantone et al. 2024, Leevy et al. 2021, Neto et al.
2023, Vinayakumar et al. 2019, Layeghy and Portmann 2022, D'hooge et al. 2023) differ only in their
feature sets, preprocessing, learners, hyperparameters and protocols. `TabularBaseline` implements the
shared pipeline once; each study subclasses it with its own config defaults, default feature list,
reported results and protocol.

Pipeline of fit
    1. features: `config.features` if given, else the study's default list for the frame's columns,
       minus `config.drop`, minus the label column
    2. labels: binary (1 attack, 0 benign, -1 unknown) or multi-class codes over `config.classes`; rows
       with unknown labels are not used for training
    3. non-finite rows: dropped (config.nonfinite = "drop") or repaired with training statistics
       ("repair"), AS-532
    4. ColumnPipeline (numeric coercion, finite repair, one-hot of `config.categorical`, scaler) fitted on
       the training rows
    5. learner (estimators.TabularModel) fitted on the design matrix

Outputs of predict
    detection   score = P(attack): p[:, 1] for binary tasks, 1 - P(benign class) for multi-class tasks
                (AS-530); label as above; unit `config.unit`; threshold `config.threshold` (0.5 is the
                argmax rule of a probabilistic classifier for two classes)
    component   class_probs [n, C] and class_label [n] for multi-class tasks; feature_importance [d]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd

from nagahana.baselines.published.base import BaselineConfig, PredictionParts, PublishedBaseline
from nagahana.baselines.published.estimators import LEARNERS, TabularModel
from nagahana.baselines.published.frames import BENIGN_TOKENS, binary_labels, build_meta, normalise_token
from nagahana.baselines.published.preprocessing import ColumnPipeline, ScalerName, numeric_matrix
from nagahana.baselines.published.protocols import SplitProtocol
from nagahana.core.errors import InvariantViolation
from nagahana.baselines.published.stages import attack_stages, class_stage_matrix, label_stage_codes
from nagahana.evaluation.predictions import DetectionPredictions, StagePredictions

Unit = Literal["flow", "window", "state_update"]


@dataclass
class TabularConfig(BaselineConfig):
    """Settings of a tabular reproduction.

    Attributes
    ----------
    learner:
        Key of estimators.LEARNERS.
    params:
        Learner hyperparameters (the paper's values or the recorded assumptions).
    features:
        Explicit feature columns; empty means the study's default list for the frame.
    drop:
        Columns removed from the feature list.
    categorical:
        Feature columns one-hot encoded.
    max_categories:
        Cap on indicator columns per categorical feature (None: every training category).
    scaler:
        "none", "minmax", "standard" or "l2" (preprocessing.py).
    task:
        "binary" or "multiclass".
    label_column, label_style:
        Label column and its coding (frames.binary_labels styles for binary tasks).
    classes:
        Class names in code order for multi-class tasks; empty learns the sorted training classes with
        the benign class first.
    family_column:
        Column naming the attack family of each row for the metadata (default: the label text).
    nonfinite:
        "drop" or "repair" training rows with non-finite feature values.
    drop_constant:
        Remove features whose finite training values are all equal (they carry no information and some
        studies remove them as invalid fields).
    threshold:
        Operating threshold on P(attack).
    unit:
        Detection unit of the produced records.
    stage_dataset:
        For multi-class tasks: the dataset whose label table (data/labels.py) maps the raw labels to ATT&CK
        tactics; when set, predict also returns stage posteriors (stages.py, AS-563).
    """

    learner: str = "random_forest"
    params: dict[str, Any] = field(default_factory=dict)
    features: tuple[str, ...] = ()
    drop: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    max_categories: int | None = None
    scaler: ScalerName = "none"
    task: Literal["binary", "multiclass"] = "binary"
    label_column: str = "Label"
    label_style: str = "auto"
    classes: tuple[str, ...] = ()
    family_column: str = ""
    nonfinite: Literal["drop", "repair"] = "drop"
    drop_constant: bool = False
    threshold: float = 0.5
    unit: Unit = "flow"
    stage_dataset: str = ""

    def validate(self) -> None:
        super().validate()
        if self.learner not in LEARNERS:
            raise ValueError(f"unknown learner {self.learner!r}; known: {sorted(LEARNERS)}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        if self.max_categories is not None and self.max_categories < 1:
            raise ValueError("max_categories must be >= 1")
        if self.task == "multiclass" and len(self.classes) == 1:
            raise ValueError("a multi-class task needs at least two classes")
        if self.label_style not in ("auto", "numeric", "text", "ctu13"):
            raise ValueError(f"unknown label_style {self.label_style!r}")


class TabularBaseline(PublishedBaseline):
    """Shared implementation of the tabular reproductions (see the module docstring)."""

    config_type: ClassVar[type[BaselineConfig]] = TabularConfig
    config: TabularConfig

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config)
        self.pipeline: ColumnPipeline | None = None
        self.model: TabularModel | None = None
        self.features_: tuple[str, ...] = ()
        self.classes_: tuple[str, ...] = ()
        self.class_stage_: np.ndarray | None = None

    def default_features(self, columns: Sequence[str]) -> tuple[str, ...]:
        """The study's feature list for a frame with these columns (default: every non-label column)."""
        cfg = self.config
        skip = {cfg.label_column, cfg.family_column, "time", "dataset", "network", "family", "novelty", "split", "entity"}
        return tuple(c for c in columns if c not in skip)

    @classmethod
    def paper_protocol(cls) -> SplitProtocol:
        """The split protocol of the paper (subclasses override)."""
        raise InvariantViolation(f"{cls.__name__} defines no paper protocol")

    def required_columns(self) -> tuple[str, ...]:
        return tuple(self.config.features)

    def _choose_params(self, train: pd.DataFrame, codes: np.ndarray, numeric: tuple[str, ...],
                       categorical: tuple[str, ...]) -> dict[str, Any]:
        """Learner hyperparameters for the final fit (default: the config's; tuning studies override)."""
        return dict(self.config.params)

    def _feature_list(self, frame: pd.DataFrame) -> tuple[str, ...]:
        cfg = self.config
        base = cfg.features if cfg.features else self.default_features(list(frame.columns))
        feats = tuple(c for c in base if c not in set(cfg.drop) and c != cfg.label_column)
        missing = [c for c in feats if c not in frame.columns]
        if missing:
            raise InvariantViolation(f"{self.spec.name}: frame lacks feature columns {missing}")
        if not feats:
            raise InvariantViolation(f"{self.spec.name}: empty feature list")
        return feats

    def _class_names(self, labels: pd.Series) -> tuple[str, ...]:
        # Learned class list: benign first, then the other classes sorted by their text.
        names = sorted({str(v).strip() for v in labels.tolist() if normalise_token(v) not in ("", "nan")})
        benign = [n for n in names if normalise_token(n) in BENIGN_TOKENS]
        return tuple(benign + [n for n in names if n not in benign])

    def _codes(self, frame: pd.DataFrame) -> np.ndarray:
        """Training / evaluation codes: binary 1/0/-1, or class indices (-1 for unknown)."""
        cfg = self.config
        if cfg.label_column not in frame.columns:
            return np.full(len(frame), -1, dtype=np.int64)
        col = frame[cfg.label_column]
        if cfg.task == "binary":
            return binary_labels(col, style=cfg.label_style)
        index = {normalise_token(c): i for i, c in enumerate(self.classes_)}
        return np.asarray([index.get(normalise_token(v), -1) for v in col.tolist()], dtype=np.int64)

    def _benign_index(self) -> int:
        for i, c in enumerate(self.classes_):
            if normalise_token(c) in BENIGN_TOKENS:
                return i
        raise InvariantViolation(f"{self.spec.name}: no benign class among {self.classes_}")

    def _family(self, frame: pd.DataFrame, codes: np.ndarray) -> np.ndarray:
        cfg = self.config
        if cfg.family_column and cfg.family_column in frame.columns:
            raw = frame[cfg.family_column].astype(str).to_numpy()
        elif cfg.label_column in frame.columns and not pd.api.types.is_numeric_dtype(frame[cfg.label_column]):
            raw = frame[cfg.label_column].astype(str).to_numpy()
        else:
            raw = np.where(codes == 1, "attack", np.where(codes == 0, "benign", "unknown")).astype(object)
        return np.asarray(["benign" if normalise_token(v) in BENIGN_TOKENS else str(v).strip() for v in raw], dtype=object)

    def _fit(self, data: pd.DataFrame, validation: pd.DataFrame | None, rng: np.random.Generator) -> None:
        cfg = self.config
        self.features_ = self._feature_list(data)
        if cfg.task == "multiclass":
            self.classes_ = tuple(cfg.classes) if cfg.classes else self._class_names(data[cfg.label_column])
        else:
            self.classes_ = ("benign", "attack")
        codes = self._codes(data)
        keep = codes >= 0
        unknown = int((~keep).sum())
        categorical = tuple(c for c in cfg.categorical if c in self.features_)
        numeric = tuple(c for c in self.features_ if c not in categorical)
        pipeline = ColumnPipeline(numeric, categorical, cfg.scaler, cfg.max_categories)
        nonfinite = pipeline.nonfinite_rows(data)
        dropped = 0
        if cfg.nonfinite == "drop":
            dropped = int((keep & nonfinite).sum())
            keep &= ~nonfinite
        train = data.loc[keep].reset_index(drop=True)
        if len(train) == 0:
            raise InvariantViolation(f"{self.spec.name}: no training rows left after label and non-finite filtering")
        constant: list[str] = []
        if cfg.drop_constant:
            # A numeric feature whose finite training values are all equal carries no information.
            x_num = numeric_matrix(train, numeric)
            for j, c in enumerate(numeric):
                fin = x_num[:, j][np.isfinite(x_num[:, j])]
                if fin.size == 0 or np.all(fin == fin[0]):
                    constant.append(c)
            if len(constant) == len(self.features_):
                raise InvariantViolation(f"{self.spec.name}: every feature is constant on the training rows")
            self.features_ = tuple(c for c in self.features_ if c not in constant)
            numeric = tuple(c for c in numeric if c not in constant)
            pipeline = ColumnPipeline(numeric, categorical, cfg.scaler, cfg.max_categories)
        params = self._choose_params(train, codes[keep], numeric, categorical)
        pipeline.fit(train)
        x = pipeline.transform(train)
        model = TabularModel(cfg.learner, params, cfg.seed, len(self.classes_), needed_by=self.spec.name)
        model.fit(x, codes[keep])
        self.pipeline, self.model = pipeline, model
        self.fit_report["params"] = dict(params)
        self.class_stage_ = None
        if cfg.task == "multiclass" and cfg.stage_dataset:
            # P(tactic | class) from the training rows' raw labels (stages.py, AS-563).
            stage = label_stage_codes(train[cfg.label_column].tolist(), cfg.stage_dataset)
            self.class_stage_ = class_stage_matrix(codes[keep], stage, len(self.classes_))
        self.fit_report.update({"rows": len(data), "train_rows": len(train), "unknown_label_rows": unknown,
                                "nonfinite_rows_dropped": dropped, "constant_features_dropped": constant,
                                "features": list(self.features_),
                                "design_width": int(x.shape[1]), "classes": list(self.classes_)})

    def _predict(self, data: pd.DataFrame) -> PredictionParts:
        assert self.pipeline is not None and self.model is not None
        cfg = self.config
        x = self.pipeline.transform(data)
        probs = self.model.predict_proba(x)
        codes = self._codes(data)
        component: dict[str, np.ndarray] = {}
        if cfg.task == "binary":
            score, label = probs[:, 1], codes
        else:
            b = self._benign_index()
            score = 1.0 - probs[:, b]
            label = np.where(codes < 0, -1, (codes != b).astype(np.int64))
            component["class_probs"] = probs
            component["class_label"] = codes
        imp = self.model.feature_importances()
        if imp is not None:
            component["feature_importance"] = imp
        meta = build_meta(data, family=self._family(data, label))
        det = DetectionPredictions(score=np.clip(score, 0.0, 1.0), label=label, unit=cfg.unit, meta=meta, threshold=cfg.threshold)
        stage = None
        if self.class_stage_ is not None:
            raw = data[cfg.label_column].tolist() if cfg.label_column in data.columns else [""] * len(data)
            stage_label = label_stage_codes(raw, cfg.stage_dataset) if cfg.label_column in data.columns else np.full(len(data), -1)
            stage = StagePredictions(probs=probs @ self.class_stage_, label=stage_label, stage_names=attack_stages(), meta=meta)
        return PredictionParts(detection=det, stage=stage, component=component)

    def _export_state(self, directory: Path) -> dict[str, Any]:
        assert self.pipeline is not None and self.model is not None
        return {"pipeline": self.pipeline.state(), "model": self.model.save(directory, "model"),
                "features": list(self.features_), "classes": list(self.classes_),
                "class_stage": None if self.class_stage_ is None else self.class_stage_.tolist()}

    def _import_state(self, directory: Path, state: Mapping[str, Any]) -> None:
        self.pipeline = ColumnPipeline.from_state(state["pipeline"])
        self.model = TabularModel.load(directory, state["model"], needed_by=self.spec.name)
        self.features_ = tuple(state["features"])
        self.classes_ = tuple(state["classes"])
        cs = state.get("class_stage")
        self.class_stage_ = None if cs is None else np.asarray(cs, dtype=np.float64)


def grid_search(
    frame: pd.DataFrame,
    codes: np.ndarray,
    *,
    numeric: tuple[str, ...],
    categorical: tuple[str, ...],
    scaler: ScalerName,
    max_categories: int | None,
    learner: str,
    base_params: Mapping[str, Any],
    grid: Mapping[str, Sequence[Any]],
    n_classes: int,
    folds: int,
    seed: int,
    needed_by: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Exhaustive search over `grid` by stratified K-fold cross-validation on `frame`.

    Every fold fits its own ColumnPipeline on the fold's training rows, so scaling statistics never
    include the fold's held-out rows. The score is the Matthews correlation coefficient of the binary
    decision at 0.5 (macro F1 of the argmax for multi-class tasks), averaged over folds; the first
    combination in grid order wins ties. Returns the best parameters (over `base_params`) and one record
    per combination.
    """
    from itertools import product

    from nagahana.baselines.published.metrics import binary_metrics, multiclass_metrics
    from nagahana.baselines.published.protocols import StratifiedKFold

    keys = sorted(grid)
    combos = [dict(zip(keys, values, strict=True)) for values in product(*(tuple(grid[k]) for k in keys))] if keys else [{}]
    strat = pd.DataFrame({"_y": codes})
    splits = list(StratifiedKFold(n_splits=folds, repeats=1, stratify="_y").splits(strat, seed))
    table: list[dict[str, Any]] = []
    best: tuple[float, int] | None = None
    for ci, combo in enumerate(combos):
        params = {**dict(base_params), **combo}
        scores: list[float] = []
        for sp in splits:
            tr, te = frame.iloc[sp.train].reset_index(drop=True), frame.iloc[sp.test].reset_index(drop=True)
            if np.unique(codes[sp.train]).size < 2:
                continue
            pipe = ColumnPipeline(numeric, categorical, scaler, max_categories).fit(tr)
            model = TabularModel(learner, params, seed, n_classes, needed_by=needed_by).fit(pipe.transform(tr), codes[sp.train])
            p = model.predict_proba(pipe.transform(te))
            if n_classes == 2:
                s = binary_metrics(p[:, 1], codes[sp.test], threshold=0.5)["mcc"]
            else:
                s = multiclass_metrics(p, codes[sp.test])["macro_f1"]
            scores.append(s if np.isfinite(s) else -1.0)
        mean = float(np.mean(scores)) if scores else -np.inf
        table.append({"params": combo, "score": mean})
        if best is None or mean > best[0]:
            best = (mean, ci)
    assert best is not None
    return {**dict(base_params), **combos[best[1]]}, table
