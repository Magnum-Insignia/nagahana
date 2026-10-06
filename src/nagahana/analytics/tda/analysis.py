"""TDA reports: point clouds, time series and the contact topology of traffic windows.

Point clouds
------------
A cloud larger than `cloud_max_points` is reduced to maxmin landmarks (after a seeded uniform sample
of `cloud_sample_rows` rows when it is larger still), and the 2 eps stability bound of the landmark
diagrams is reported (`complex.maxmin_landmarks`). The Rips radius is chosen by the first rule that
applies: `max_radius`; the `radius_quantile` quantile of pairwise distances; the degree budget, the
smallest radius at which the mean vertex degree reaches `max_mean_degree` (capped by the enclosing
radius, beyond which nothing changes); the rule used is part of the report, and classes alive at the
radius are reported as essential (death "inf").

Corpus clouds
-------------
Rows are state updates; features are numeric columns. Only rows where every chosen feature carries a
value enter the cloud (D-41: no imputation). Values are put on the slog1p scale (AS-31) and scaled by
their median absolute deviation, so that no single heavy-tailed field fixes the geometry.

Time series
-----------
The update-count series of a corpus (bins of `sw_bin_seconds`) is embedded with a sliding window
whose delay and dimension come from the average mutual information and false nearest neighbours
unless given, and scored for periodicity (`embedding.periodicity_score`).

Contact topology of windows
---------------------------
For each window (the corpus window records, or consecutive slices of `window_seconds`), the contact
filtration of `complex.contact_complex` is built from the window's updates: an initiator-responder
pair enters at its first contact time (relative to the window start) and each entity at its first
appearance. H_0 records how components merge as contacts accumulate, H_1 the cycles of communication.
The bottleneck and Wasserstein distances between the diagrams of consecutive windows form a
topological drift series of the traffic structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from nagahana.analytics.config import TDAConfig, to_dict
from nagahana.analytics.dependence import slog1p
from nagahana.analytics.report import Figure, Report
from nagahana.analytics.robust import MAD_NORMAL
from nagahana.analytics.tda import complex as cplx
from nagahana.analytics.tda import diagrams as dg
from nagahana.analytics.tda import embedding as emb
from nagahana.analytics.tda import vectorize as vec
from nagahana.analytics.tda.homology import Persistence, persistence

if TYPE_CHECKING:                                                        # the data model is needed by corpus inputs only
    from nagahana.analytics.corpus import Corpus


def choose_radius(points: np.ndarray, cfg: TDAConfig) -> tuple[float, str]:
    """(radius, rule) by the module-docstring rules."""
    if cfg.max_radius is not None:
        return float(cfg.max_radius), "max_radius"
    name = {"euclidean": "euclidean", "chebyshev": "chebyshev", "cityblock": "cityblock"}[cfg.metric]
    n = points.shape[0]
    if n < 2:
        return 0.0, "single_point"
    dists = pdist(points, metric=name)
    if cfg.radius_quantile is not None:
        if not 0.0 < cfg.radius_quantile <= 1.0:
            raise ValueError("radius_quantile must be in (0, 1]")
        return float(np.quantile(dists, cfg.radius_quantile)), "radius_quantile"
    enclosing = cplx.enclosing_radius(points, metric=cfg.metric)
    edges = int(min(dists.size, max(1, round(cfg.max_mean_degree * n / 2.0))))
    r_deg = float(np.partition(dists, edges - 1)[edges - 1])
    return (r_deg, "degree_budget") if r_deg < enclosing else (enclosing, "enclosing_radius")


def _landmarks(points: np.ndarray, cfg: TDAConfig) -> tuple[np.ndarray, np.ndarray, float, str]:
    """(cloud, indices into points, covering radius, note) after the landmark rules of the module docstring."""
    n = points.shape[0]
    target = cfg.n_landmarks if cfg.n_landmarks is not None else (cfg.cloud_max_points if n > cfg.cloud_max_points else None)
    if target is None or target >= n:
        return points, np.arange(n), 0.0, "all points"
    rng = np.random.default_rng(cfg.seed)
    base = np.sort(rng.choice(n, size=cfg.cloud_sample_rows, replace=False)) if n > cfg.cloud_sample_rows else np.arange(n)
    idx, eps = cplx.maxmin_landmarks(points[base], int(target), metric=cfg.metric)
    note = (f"{int(target)} maxmin landmarks of a seeded sample of {base.size} points" if base.size < n
            else f"{int(target)} maxmin landmarks")
    return points[base][idx], base[idx], eps, note


def diagram_frame(diagrams: tuple[dg.Diagram, ...] | list[dg.Diagram], **keys: object) -> pd.DataFrame:
    """Long table of diagram points: (keys..., dim, birth, death, persistence)."""
    rows = [pd.DataFrame({**{k: v for k, v in keys.items()}, "dim": d.dim, "birth": d.birth, "death": d.death,
                          "persistence": d.persistence}) for d in diagrams]
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if out.empty:
        out = pd.DataFrame(columns=[*keys, "dim", "birth", "death", "persistence"])
    return out


def _vectorisations(per: Persistence, cfg: TDAConfig, cap: float) -> dict[str, pd.DataFrame]:
    """Betti curves, landscapes and persistence images of every dimension as long tables."""
    diags = [d.filtered(cfg.min_persistence) for d in per.diagrams]
    grid = vec.grid_for(diags, cfg.betti_resolution, cap=cap)
    betti = pd.concat([pd.DataFrame({"dim": d.dim, "t": grid, "betti": vec.betti_curve(d, grid)}) for d in diags],
                      ignore_index=True)
    lgrid = vec.grid_for(diags, cfg.landscape_resolution, cap=cap)
    land = []
    for d in diags:
        lv = vec.landscape(d, lgrid, k=cfg.landscape_k, cap=cap)
        for j in range(lv.shape[0]):
            land.append(pd.DataFrame({"dim": d.dim, "level": j + 1, "t": lgrid, "value": lv[j]}))
    images = []
    for d in diags:
        c = d.capped(cap) if len(d) else d
        b, de = c.finite()
        if b.size == 0:
            continue
        pmax = float((de - b).max())
        sigma = cfg.image_sigma if cfg.image_sigma is not None else max(pmax / 30.0, 1e-12)
        img = vec.persistence_image(c, resolution=cfg.image_resolution, sigma=sigma,
                                    birth_range=(float(b.min()), float(b.max()) + 1e-12), pers_range=(0.0, pmax))
        yy, xx = np.meshgrid(np.arange(cfg.image_resolution), np.arange(cfg.image_resolution), indexing="ij")
        images.append(pd.DataFrame({"dim": d.dim, "pers_bin": yy.ravel(), "birth_bin": xx.ravel(), "value": img.ravel()}))
    return {"betti": betti, "landscapes": pd.concat(land, ignore_index=True) if land else pd.DataFrame(
        columns=["dim", "level", "t", "value"]), "images": pd.concat(images, ignore_index=True) if images else pd.DataFrame(
        columns=["dim", "pers_bin", "birth_bin", "value"])}


def point_cloud_report(points: np.ndarray, cfg: TDAConfig | None = None, *, title: str = "Topology of a point cloud",
                       kind: str = "tda") -> Report:
    """Persistent homology of a point cloud with its vectorisations (module docstring)."""
    cfg = cfg or TDAConfig()
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or not np.isfinite(x).all():
        raise ValueError("points must be a non-empty finite [n, d] array")
    cloud, idx, eps, note = _landmarks(x, cfg)
    radius, rule = choose_radius(cloud, cfg)
    cx = cplx.rips_complex(cloud, max_dim=cfg.max_dimension, max_radius=radius, metric=cfg.metric,
                           max_simplices=cfg.max_simplices)
    per = persistence(cx, max_dim=cfg.max_dimension, algorithm=cfg.algorithm, clearing=cfg.clearing,
                      apparent_pairs=cfg.apparent_pairs)
    rep = Report(kind=kind, title=title, provenance={"config": to_dict(cfg)})
    rep.summary.update({
        "points": int(x.shape[0]), "cloud_points": int(cloud.shape[0]), "landmarks": note, "covering_radius": eps,
        "bottleneck_bound_from_landmarks": 2.0 * eps, "radius": radius, "radius_rule": rule,
        "simplices_per_dimension": cx.counts(), **{f"homology.{k}": v for k, v in per.stats.items()},
    })
    diags = [d.filtered(cfg.min_persistence) for d in per.diagrams]
    summaries = pd.DataFrame([dg.summary(d) for d in diags])
    rep.add_table("diagram", diagram_frame(diags), title="Persistence diagram",
                  description="One row per bar; death 'inf' = alive at the radius (truncated filtration).")
    rep.add_table("diagram_summary", summaries, title="Diagram summaries per dimension")
    for _, row in summaries.iterrows():
        rep.summary[f"H{int(row['dim'])}.bars"] = int(row["count"])
        rep.summary[f"H{int(row['dim'])}.essential_at_radius"] = int(row["essential"])
        rep.summary[f"H{int(row['dim'])}.max_finite_persistence"] = float(row["max_persistence"])
    cap = radius if np.isfinite(radius) else float(max((np.max(d.death[np.isfinite(d.death)], initial=0.0) for d in diags),
                                                        default=1.0))
    vecs = _vectorisations(per, cfg, cap)
    frame = diagram_frame(diags)
    plot = frame.assign(death_plot=np.where(np.isfinite(frame["death"].astype(float)), frame["death"], cap))
    rep.add_figure(Figure(name="diagram", kind="diagram", data=plot, x="birth", y="death_plot", series="dim",
                          title="Persistence diagram", x_label="birth", y_label="death",
                          description="Essential bars are drawn at the radius (death_plot)."))
    rep.add_figure(Figure(name="betti_curves", kind="step", data=vecs["betti"], x="t", y="betti", series="dim",
                          title="Betti curves"))
    rep.add_figure(Figure(name="landscapes", kind="line", data=vecs["landscapes"], x="t", y="value", series="level",
                          title="Persistence landscapes", description="Column 'dim' separates dimensions."))
    if len(vecs["images"]):
        rep.add_figure(Figure(name="persistence_images", kind="heatmap", data=vecs["images"], x="birth_bin", y="pers_bin",
                              value="value", title="Persistence images", description="Column 'dim' separates dimensions."))
    if eps > 0:
        rep.notes.append(f"Landmark diagrams are within bottleneck distance {2 * eps:.6g} of the diagrams of the full "
                         "cloud (2 x covering radius; Chazal et al. 2009).")
    return rep


def corpus_cloud(corpus: Corpus, cfg: TDAConfig) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(cloud [m, f], feature names, corpus rows) of the module docstring ("Corpus clouds")."""
    contributing = corpus.contributing()
    numeric = corpus.numeric_columns()
    if cfg.features:
        cols = [corpus.column_index(c) for c in cfg.features]
    else:
        share = contributing.mean(axis=0) if len(corpus) else np.zeros(len(corpus.columns))
        cols = [j for j in range(len(corpus.columns)) if numeric[j] and share[j] >= 0.5]
    if not cols:
        raise ValueError("no numeric feature carries values in at least half of the rows; name features explicitly")
    ci = np.asarray(cols, dtype=np.int64)
    rows = np.flatnonzero(contributing[:, ci].all(axis=1))
    z = slog1p(corpus.values[np.ix_(rows, ci)])
    med = np.median(z, axis=0)
    mad = np.median(np.abs(z - med), axis=0) * MAD_NORMAL
    sd = z.std(axis=0)
    scale = np.where(mad > 0, mad, np.where(sd > 0, sd, 1.0))
    return (z - med) / scale, [corpus.columns[j].name for j in cols], rows


def series_report(series: np.ndarray, cfg: TDAConfig | None = None, *, bin_seconds: float | None = None) -> Report:
    """Sliding-window periodicity of a series (module docstring, "Time series")."""
    cfg = cfg or TDAConfig()
    x = np.asarray(series, dtype=np.float64).reshape(-1)
    if x.size > cfg.sw_max_series:
        factor = int(np.ceil(x.size / cfg.sw_max_series))
        x = np.add.reduceat(x, np.arange(0, x.size, factor))
        bin_seconds = None if bin_seconds is None else bin_seconds * factor
    rep = Report(kind="tda_series", title="Sliding-window topology of a series", provenance={"config": to_dict(cfg)})
    max_delay = int(min(cfg.sw_max_delay, max(1, x.size // 4)))
    if cfg.sw_delay is None:
        delay, ami = emb.delay_by_ami(x, max_delay=max_delay)
        rep.add_figure(Figure(name="ami", kind="line", data=pd.DataFrame({"delay": np.arange(ami.size), "ami": ami}),
                              x="delay", y="ami", title="Average mutual information"))
    else:
        delay = int(cfg.sw_delay)
    if cfg.sw_dimension is None:
        dim, fnn = emb.dimension_by_fnn(x, delay=delay, max_dimension=cfg.sw_max_dimension)
        rep.add_figure(Figure(name="fnn", kind="line", data=pd.DataFrame({"dimension": np.arange(1, fnn.size + 1),
                                                                          "false_share": fnn}),
                              x="dimension", y="false_share", title="False nearest neighbours"))
    else:
        dim = int(cfg.sw_dimension)
    dim = max(dim, 2)
    per = emb.periodicity_score(x, dimension=dim, delay=delay, stride=cfg.sw_stride, max_points=cfg.sw_max_points)
    rep.summary.update({"length": int(x.size), "delay": delay, "dimension": dim, "window_points": per.points,
                        "window_covering_radius": per.covering_radius,
                        "periodicity_score": per.score, "max_h1_persistence": per.max_persistence,
                        "bin_seconds": bin_seconds if bin_seconds is not None else float("nan")})
    if bin_seconds is not None:
        rep.summary["window_length_s"] = (dim - 1) * delay * bin_seconds
    rep.add_table("h1_diagram", diagram_frame([per.diagram]), title="H1 of the normalised window cloud")
    return rep


def _window_slices(corpus: Corpus, cfg: TDAConfig) -> list[tuple[str, np.ndarray]]:
    """(window id, rows) for the contact-topology windows (module docstring)."""
    if (corpus.window != "").any():
        frame = pd.DataFrame({"w": corpus.window, "t": np.nan_to_num(corpus.time, nan=0.0)})
        frame = frame[frame["w"] != ""]
        order = frame.groupby("w")["t"].min().sort_values(kind="stable").index
        groups = frame.groupby("w").indices
        out = [(str(w), np.asarray(groups[w])) for w in order]
    else:
        t = corpus.time
        if not np.isfinite(t).any():
            return [("all", np.arange(len(corpus)))]
        b = np.floor((t - np.nanmin(t)) / cfg.window_seconds).astype(np.int64)
        out = [(f"bin{k}", np.flatnonzero(b == k)) for k in np.unique(b[b >= 0]).tolist()]
    if len(out) > cfg.max_windows:
        keep = np.unique(np.linspace(0, len(out) - 1, cfg.max_windows).round().astype(np.int64))
        out = [out[i] for i in keep]
    return out


def window_topology(corpus: Corpus, cfg: TDAConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[list[dg.Diagram]]]:
    """(per-window summaries, consecutive-window distances, diagrams) of the contact filtrations."""
    max_dim = min(cfg.max_dimension, 1)
    rows_out, diagrams = [], []
    for wid, rows in _window_slices(corpus, cfg):
        e = corpus.entities[rows, :2]
        t = corpus.time[rows]
        t = np.zeros(rows.size) if not np.isfinite(t).any() else t - np.nanmin(t)
        ok = (e[:, 0] >= 0) & (e[:, 1] >= 0) & (e[:, 0] != e[:, 1])
        ents, inv = np.unique(e[ok], return_inverse=True)
        v = ents.size
        if v < 2:
            diagrams.append([dg.Diagram(k, np.zeros(0), np.zeros(0)) for k in range(max_dim + 1)])
            rows_out.append({"window": wid, "updates": int(rows.size), "entities": int(v), "contacts": 0})
            continue
        pair = inv.reshape(-1, 2)
        c1 = np.full((v, v), np.inf)
        tt = t[ok]
        np.minimum.at(c1, (pair[:, 0], pair[:, 1]), tt)
        c1 = np.minimum(c1, c1.T)
        np.fill_diagonal(c1, 0.0)
        cx = cplx.contact_complex(c1, max_dim=max_dim, max_simplices=cfg.max_simplices)
        per = persistence(cx, max_dim=max_dim, algorithm=cfg.algorithm, clearing=cfg.clearing,
                          apparent_pairs=cfg.apparent_pairs)
        diagrams.append(list(per.diagrams))
        row: dict[str, object] = {"window": wid, "updates": int(rows.size), "entities": int(v),
                                  "contacts": int(np.isfinite(c1[np.triu_indices(v, 1)]).sum())}
        for d in per.diagrams:
            s = dg.summary(d)
            row[f"H{d.dim}_bars"] = int(s["count"])
            row[f"H{d.dim}_essential"] = int(s["essential"])
            row[f"H{d.dim}_total_persistence"] = s["total_persistence"]
        rows_out.append(row)
    dist_rows = []
    for i in range(1, len(diagrams)):
        row = {"window": rows_out[i]["window"], "previous": rows_out[i - 1]["window"]}
        for k in range(max_dim + 1):
            a, b = diagrams[i - 1][k], diagrams[i][k]
            row[f"H{k}_bottleneck_finite"] = dg.bottleneck_distance(dg.Diagram(k, *a.finite()), dg.Diagram(k, *b.finite()))
            row[f"H{k}_wasserstein_finite"] = dg.wasserstein_distance(dg.Diagram(k, *a.finite()), dg.Diagram(k, *b.finite()),
                                                                      p=cfg.wasserstein_p, norm=cfg.ground_norm)
            row[f"H{k}_essential_change"] = int(b.essential().size) - int(a.essential().size)
        dist_rows.append(row)
    return pd.DataFrame(rows_out), pd.DataFrame(dist_rows), diagrams


def corpus_report(corpus: Corpus, cfg: TDAConfig | None = None) -> Report:
    """Point-cloud topology of the state updates, periodicity of the update counts, contact topology of windows."""
    cfg = cfg or TDAConfig()
    rep = Report(kind="tda", title="Topological data analysis", provenance={"config": to_dict(cfg)})
    cloud, names, rows = corpus_cloud(corpus, cfg)
    rep.summary["cloud_features"] = names
    rep.summary["cloud_rows_complete"] = int(rows.size)
    if rows.size >= 2:
        rep.merge(point_cloud_report(cloud, cfg, title="State updates as a point cloud"), prefix="cloud")
    else:
        rep.notes.append("Fewer than two rows carry every cloud feature; the point-cloud topology was skipped.")
    t = corpus.time[np.isfinite(corpus.time)]
    if t.size >= 8:
        b = np.floor((t - t.min()) / cfg.sw_bin_seconds).astype(np.int64)
        counts = np.bincount(b).astype(np.float64)
        if counts.size >= 16:
            rep.merge(series_report(counts, cfg, bin_seconds=cfg.sw_bin_seconds), prefix="periodicity")
        else:
            rep.notes.append("The update-count series is shorter than 16 bins; periodicity was skipped.")
    else:
        rep.notes.append("The corpus carries fewer than 8 event times; periodicity was skipped.")
    summ, dist, _ = window_topology(corpus, cfg)
    rep.add_table("window_topology", summ, title="Contact topology per window",
                  description="Contact filtration by first contact time (relative to the window start).")
    rep.add_table("topological_drift", dist, title="Topological drift between consecutive windows",
                  description="Bottleneck and Wasserstein distances of the finite parts; change in essential classes.")
    if len(dist):
        col = "H0_bottleneck_finite"
        rep.add_figure(Figure(name="topological_drift", kind="line", data=dist.assign(step=np.arange(len(dist))),
                              x="step", y=col, title="Bottleneck distance of H0 between consecutive windows"))
    return rep


__all__ = ["choose_radius", "corpus_cloud", "corpus_report", "diagram_frame", "point_cloud_report", "series_report",
           "window_topology"]
