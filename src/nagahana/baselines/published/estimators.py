"""The classical learners of the reproduced studies behind one interface.

`TabularModel` wraps a scikit-learn, XGBoost or LightGBM classifier selected by name and gives every
reproduction the same four operations: fit on an integer-coded label vector, `predict_proba` with one
column per class of the task (classes absent from the training data get probability 0), feature
importances where the learner defines them, and persistence.

Seeds (AS-533)
--------------
The seed is passed to every learner that has a random_state (forests, boosting, MLP, SVC's probability
calibration, CatBoost's random_seed alias). LightGBM additionally runs with deterministic=True and
force_row_wise=True, which its documentation states makes results stable for the same data and
parameters. XGBoost's "hist" tree method is deterministic for a fixed seed on CPU. CatBoost runs silent
and writes no training-log directory (verbose=False, allow_writing_files=False), which changes nothing
in the fitted model.

Persistence (AS-534)
--------------------
XGBoost, LightGBM and CatBoost models are written in their native formats (JSON model file, text model
file, CatBoost binary .cbm), which stay readable across library versions. scikit-learn estimators have
no native format and are pickled; the manifest of base.py records the SHA-256 of the file and `load`
refuses a file whose digest differs, so only files this package wrote are ever unpickled.

Scores of margin classifiers
----------------------------
A learner without predict_proba (LinearSVC) reports sigma(decision_function) for the positive class:
a monotone score in (0, 1), not a calibrated probability. The decision rule of the learner (margin > 0)
is the threshold 0.5 of that score, so the operating point is preserved exactly.
"""

from __future__ import annotations

import inspect
import pickle
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nagahana.baselines.published.backends import require, version_of
from nagahana.core.errors import InvariantViolation


@dataclass(frozen=True)
class LearnerSpec:
    """How to build one learner: import path, fixed parameters and library."""

    module: str
    cls: str
    library: str
    fixed: Mapping[str, Any] = field(default_factory=dict)
    title: str = ""


#: Learners used by the reproduced studies, by name.
LEARNERS: dict[str, LearnerSpec] = {
    "extra_trees": LearnerSpec("sklearn.ensemble", "ExtraTreesClassifier", "sklearn", title="extra trees"),
    "random_forest": LearnerSpec("sklearn.ensemble", "RandomForestClassifier", "sklearn", title="random forest"),
    "decision_tree": LearnerSpec("sklearn.tree", "DecisionTreeClassifier", "sklearn", title="decision tree"),
    "gradient_boosting": LearnerSpec("sklearn.ensemble", "GradientBoostingClassifier", "sklearn", title="gradient boosting"),
    "adaboost": LearnerSpec("sklearn.ensemble", "AdaBoostClassifier", "sklearn", title="AdaBoost"),
    "lda": LearnerSpec("sklearn.discriminant_analysis", "LinearDiscriminantAnalysis", "sklearn",
                       title="linear discriminant analysis"),
    "qda": LearnerSpec("sklearn.discriminant_analysis", "QuadraticDiscriminantAnalysis", "sklearn",
                       title="quadratic discriminant analysis"),
    "logistic_regression": LearnerSpec("sklearn.linear_model", "LogisticRegression", "sklearn", title="logistic regression"),
    "gaussian_nb": LearnerSpec("sklearn.naive_bayes", "GaussianNB", "sklearn", title="Gaussian naive Bayes"),
    "knn": LearnerSpec("sklearn.neighbors", "KNeighborsClassifier", "sklearn", title="k nearest neighbours"),
    "svm_rbf": LearnerSpec("sklearn.svm", "SVC", "sklearn", {"kernel": "rbf", "probability": True}, title="SVM (RBF kernel)"),
    "linear_svm": LearnerSpec("sklearn.svm", "LinearSVC", "sklearn", title="linear SVM"),
    "mlp": LearnerSpec("sklearn.neural_network", "MLPClassifier", "sklearn", title="multi-layer perceptron"),
    "isolation_forest": LearnerSpec("sklearn.ensemble", "IsolationForest", "sklearn", title="isolation forest"),
    "xgboost": LearnerSpec("xgboost", "XGBClassifier", "xgboost", {"tree_method": "hist"}, title="XGBoost"),
    "lightgbm": LearnerSpec("lightgbm", "LGBMClassifier", "lightgbm",
                            {"deterministic": True, "force_row_wise": True, "verbose": -1}, title="LightGBM"),
    "catboost": LearnerSpec("catboost", "CatBoostClassifier", "catboost",
                            {"verbose": False, "allow_writing_files": False}, title="CatBoost"),
}

#: Parameter names under which a learner accepts its seed (CatBoost: random_seed or its alias random_state).
_SEED_KEYS: tuple[str, ...] = ("random_state", "random_seed", "seed")


def build_learner(name: str, params: Mapping[str, Any], seed: int, *, needed_by: str) -> Any:
    """Instantiate learner `name` with `params` over the fixed parameters, seeded where it accepts a seed."""
    if name not in LEARNERS:
        raise ValueError(f"unknown learner {name!r}; known: {sorted(LEARNERS)}")
    spec = LEARNERS[name]
    cls = getattr(require(spec.module, needed_by=needed_by), spec.cls)
    kwargs: dict[str, Any] = {**spec.fixed, **dict(params)}
    accepted = inspect.signature(cls.__init__).parameters
    # Seed the learner unless the caller set a seed under any accepted name (CatBoost rejects two).
    if not any(k in kwargs for k in _SEED_KEYS) and ("random_state" in accepted or spec.library in ("xgboost", "catboost")):
        kwargs["random_state"] = seed
    unknown = [k for k in kwargs if k not in accepted and spec.library == "sklearn"]
    if unknown:
        raise ValueError(f"{spec.cls} does not accept parameters {unknown}")
    return cls(**kwargs)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


class TabularModel:
    """A learner with integer class codes 0 ... n_classes - 1 and full-width probabilities.

    Parameters
    ----------
    learner:
        Key of LEARNERS.
    params:
        Hyperparameters passed to the learner (the paper's values or the recorded assumptions).
    seed:
        Seed for learners with a random_state.
    n_classes:
        Number of classes of the task (2 for binary detection).
    needed_by:
        Name of the reproduction, for the error message of a missing library.
    """

    def __init__(self, learner: str, params: Mapping[str, Any], seed: int, n_classes: int, *, needed_by: str) -> None:
        if n_classes < 2:
            raise ValueError("a classification task needs at least two classes")
        self.learner = learner
        self.params = dict(params)
        self.seed = int(seed)
        self.n_classes = int(n_classes)
        self.needed_by = needed_by
        self.estimator: Any = None
        self.present: np.ndarray = np.zeros(0, dtype=np.int64)
        self._booster: Any = None

    @property
    def library(self) -> str:
        return LEARNERS[self.learner].library

    def fit(self, x: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray | None = None) -> TabularModel:
        """Fit on x [n, d] and integer labels y [n] in 0 ... n_classes - 1."""
        y = np.asarray(y, dtype=np.int64)
        if x.ndim != 2 or y.shape != (x.shape[0],):
            raise InvariantViolation("x must be [n, d] and y [n]")
        if y.size == 0 or y.min() < 0 or y.max() >= self.n_classes:
            raise InvariantViolation(f"labels must lie in 0 ... {self.n_classes - 1}")
        self.present = np.unique(y)
        if self.present.size < 2:
            raise InvariantViolation(f"{self.needed_by}: training data contain a single class ({self.present.tolist()})")
        # Learners see contiguous codes over the classes present (XGBoost requires it).
        y_fit = np.searchsorted(self.present, y)
        self.estimator = build_learner(self.learner, self.params, self.seed, needed_by=self.needed_by)
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=np.float64)
        self.estimator.fit(x, y_fit, **fit_kwargs)
        self._booster = None
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Probabilities [n, n_classes]; columns of classes absent from training are 0."""
        if self.estimator is None and self._booster is None:
            raise InvariantViolation("TabularModel.predict_proba called before fit")
        k = self.present.size
        if self._booster is not None:
            raw = np.asarray(self._booster.predict(x), dtype=np.float64)
            p = np.stack([1.0 - raw, raw], axis=1) if raw.ndim == 1 else raw
        elif hasattr(self.estimator, "predict_proba"):
            p = np.asarray(self.estimator.predict_proba(x), dtype=np.float64)
        else:
            margin = np.asarray(self.estimator.decision_function(x), dtype=np.float64)
            if margin.ndim == 1:
                pos = _sigmoid(margin)
                p = np.stack([1.0 - pos, pos], axis=1)
            else:
                e = np.exp(margin - margin.max(axis=1, keepdims=True))
                p = e / e.sum(axis=1, keepdims=True)
        if p.shape != (x.shape[0], k):
            raise InvariantViolation(f"learner returned probabilities of shape {p.shape}, expected {(x.shape[0], k)}")
        out = np.zeros((x.shape[0], self.n_classes), dtype=np.float64)
        out[:, self.present] = p
        # Guard against rounding outside [0, 1] and rows that do not sum to 1.
        out = np.clip(out, 0.0, 1.0)
        s = out.sum(axis=1, keepdims=True)
        s[s == 0.0] = 1.0
        return out / s

    def feature_importances(self) -> np.ndarray | None:
        """Impurity or gain importances where the learner defines them, |coef| for linear models, else None."""
        if self._booster is not None:
            # A reloaded LightGBM model: split counts, the default importance of LGBMClassifier.
            return np.asarray(self._booster.feature_importance(importance_type="split"), dtype=np.float64)
        est = self.estimator
        if est is None:
            return None
        if hasattr(est, "feature_importances_"):
            return np.asarray(est.feature_importances_, dtype=np.float64)
        if hasattr(est, "coef_"):
            return np.abs(np.asarray(est.coef_, dtype=np.float64)).sum(axis=0)
        return None

    def save(self, directory: Path, stem: str) -> dict[str, Any]:
        """Write the fitted learner as directory/stem.<ext>; return the state needed by `load`."""
        if self.estimator is None and self._booster is None:
            raise InvariantViolation("TabularModel.save called before fit")
        directory.mkdir(parents=True, exist_ok=True)
        if self.library == "xgboost":
            fname = f"{stem}.xgb.json"
            self.estimator.save_model(str(directory / fname))
        elif self.library == "lightgbm":
            fname = f"{stem}.lgb.txt"
            booster = self._booster if self._booster is not None else self.estimator.booster_
            booster.save_model(str(directory / fname))
        elif self.library == "catboost":
            fname = f"{stem}.cbm"
            self.estimator.save_model(str(directory / fname), format="cbm")
        else:
            fname = f"{stem}.sklearn.pkl"
            (directory / fname).write_bytes(pickle.dumps(self.estimator, protocol=pickle.HIGHEST_PROTOCOL))
        return {"learner": self.learner, "params": self.params, "seed": self.seed, "n_classes": self.n_classes,
                "present": self.present.tolist(), "file": fname, "library_version": version_of(LEARNERS[self.learner].module)}

    @classmethod
    def load(cls, directory: Path, state: Mapping[str, Any], *, needed_by: str) -> TabularModel:
        """Restore a learner written by `save` (the caller has verified the file digest)."""
        obj = cls(state["learner"], state["params"], int(state["seed"]), int(state["n_classes"]), needed_by=needed_by)
        obj.present = np.asarray(state["present"], dtype=np.int64)
        path = directory / str(state["file"])
        if obj.library == "xgboost":
            xgb = require("xgboost", needed_by=needed_by)
            est = xgb.XGBClassifier()
            est.load_model(str(path))
            obj.estimator = est
        elif obj.library == "lightgbm":
            lgb = require("lightgbm", needed_by=needed_by)
            obj._booster = lgb.Booster(model_file=str(path))
            obj.estimator = None
        elif obj.library == "catboost":
            cb = require("catboost", needed_by=needed_by)
            est = cb.CatBoostClassifier(**LEARNERS["catboost"].fixed)
            est.load_model(str(path), format="cbm")
            obj.estimator = est
        else:
            require(LEARNERS[obj.learner].module, needed_by=needed_by)
            obj.estimator = pickle.loads(path.read_bytes())
        return obj
