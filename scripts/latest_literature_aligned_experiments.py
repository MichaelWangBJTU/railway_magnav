from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import axis_calibrated_full_matching as ac


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\latest_literature_aligned_experiments")
STEP_M = 0.5
WINDOWS_M = [50.0, 100.0, 150.0]
QUERY_STRIDE_M = 20.0
SLAC_CANDIDATE_STRIDE_M = 5.0


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def robust_z_window(x: np.ndarray) -> np.ndarray | None:
    x = np.asarray(x, dtype=float)
    if np.isfinite(x).sum() < max(10, int(0.8 * len(x))):
        return None
    s = pd.Series(x).interpolate(limit_direction="both").to_numpy(float)
    z = ac.robust_z(s)
    if not np.isfinite(z).all():
        return None
    return z


def candidate_rank_stats(
    starts: np.ndarray,
    scores: np.ndarray,
    true_start: float,
    tolerances_m: tuple[float, ...] = (5.0, 10.0, 25.0),
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    out: dict[str, float] = {}
    valid = np.isfinite(scores)
    if not valid.any():
        return {k: math.nan for k in ["pred_start_m", "top1_abs_error_m", "score_gap_10m"]}
    order = np.argsort(scores[valid])[::-1]
    valid_idx = np.flatnonzero(valid)
    ranked_idx = valid_idx[order]
    pred = float(starts[ranked_idx[0]])
    out["pred_start_m"] = pred
    out["top1_abs_error_m"] = abs(pred - true_start)

    far = np.abs(starts - pred) >= 10.0
    second = np.nanmax(scores[far & valid]) if np.any(far & valid) else math.nan
    out["score_gap_10m"] = float(scores[ranked_idx[0]] - second) if np.isfinite(second) else math.nan

    abs_err_ranked = np.abs(starts[ranked_idx] - true_start)
    for tol in tolerances_m:
        within = np.flatnonzero(abs_err_ranked <= tol)
        rank = int(within[0] + 1) if len(within) else math.inf
        out[f"rank_within_{tol:g}m"] = float(rank) if math.isfinite(rank) else math.nan
        for k in ks:
            out[f"top{k}_within_{tol:g}m"] = float(rank <= k)
    return out


def reference_window_cache(
    ref_dist: np.ndarray,
    ref_features: dict[str, np.ndarray],
    feature_names: list[str],
    n: int,
    candidate_stride_m: float = STEP_M,
) -> tuple[np.ndarray, list[np.ndarray]]:
    stride_n = max(1, int(round(candidate_stride_m / STEP_M)))
    starts = []
    rows_by_feat: list[list[np.ndarray]] = [[] for _ in feature_names]
    for i in range(0, len(ref_dist) - n + 1, stride_n):
        feat_windows = []
        ok = True
        for name in feature_names:
            zw = robust_z_window(ref_features[name][i : i + n])
            if zw is None:
                ok = False
                break
            feat_windows.append(zw.astype(np.float32))
        if not ok:
            continue
        starts.append(float(ref_dist[i]))
        for j, zw in enumerate(feat_windows):
            rows_by_feat[j].append(zw)
    if not starts:
        return np.array([], dtype=float), []
    return np.asarray(starts, dtype=float), [np.vstack(rows) for rows in rows_by_feat]


def ncc_or_msd_scores(
    query_window: list[np.ndarray],
    ref_starts: np.ndarray,
    ref_matrices: list[np.ndarray],
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if len(ref_starts) == 0:
        return ref_starts, np.array([], dtype=float)
    qz_windows = []
    for w in query_window:
        zw = robust_z_window(w)
        if zw is None:
            return ref_starts, np.full(len(ref_starts), np.nan)
        qz_windows.append(zw.astype(np.float32))
    scores = np.zeros(len(ref_starts), dtype=np.float32)
    for qz, mat in zip(qz_windows, ref_matrices):
        if mode == "ncc":
            scores += mat @ qz / float(len(qz))
        elif mode == "msd":
            scores += -np.mean((mat - qz[None, :]) ** 2, axis=1)
        else:
            raise ValueError(mode)
    scores /= float(len(ref_matrices))
    return ref_starts, scores.astype(float)


def affine_slac_score(q_mat: np.ndarray, r_mat: np.ndarray, ridge: float = 1e-2) -> float:
    mask = np.isfinite(q_mat).all(axis=1) & np.isfinite(r_mat).all(axis=1)
    if mask.sum() < max(30, int(0.8 * len(q_mat))):
        return math.nan
    qz_cols = []
    rz_cols = []
    for j in range(q_mat.shape[1]):
        qz = robust_z_window(q_mat[mask, j])
        rz = robust_z_window(r_mat[mask, j])
        if qz is None or rz is None:
            return math.nan
        qz_cols.append(qz)
        rz_cols.append(rz)
    qz = np.column_stack(qz_cols)
    rz = np.column_stack(rz_cols)
    x = np.column_stack([qz, np.ones(len(qz))])
    xtx = x.T @ x
    reg = ridge * np.eye(xtx.shape[0])
    reg[-1, -1] = 0.0
    try:
        beta = np.linalg.solve(xtx + reg, x.T @ rz)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(x, rz, rcond=None)[0]
    pred = x @ beta
    rmse = float(np.sqrt(np.mean((pred - rz) ** 2)))
    return -rmse


def slac_scores(
    q_windows: list[np.ndarray],
    ref_dist: np.ndarray,
    ref_features: dict[str, np.ndarray],
    ref_feature_names: list[str],
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    q_mat = np.column_stack(q_windows)
    stride_n = max(1, int(round(SLAC_CANDIDATE_STRIDE_M / STEP_M)))
    starts = []
    scores = []
    for i in range(0, len(ref_dist) - n + 1, stride_n):
        r_mat = np.column_stack([ref_features[name][i : i + n] for name in ref_feature_names])
        score = affine_slac_score(q_mat, r_mat)
        starts.append(float(ref_dist[i]))
        scores.append(score)
    return np.asarray(starts, dtype=float), np.asarray(scores, dtype=float)


def query_windows(q: ac.QueryFeature, feature_names: list[str], n: int) -> list[tuple[int, float, list[np.ndarray]]]:
    stride_n = max(1, int(round(QUERY_STRIDE_M / STEP_M)))
    out = []
    arrays = [q.features[name] for name in feature_names]
    for i in range(0, len(q.distance) - n + 1, stride_n):
        d_win = q.distance[i : i + n]
        if np.nanmax(d_win) - np.nanmin(d_win) < (n - 1) * STEP_M - STEP_M:
            continue
        windows = [arr[i : i + n] for arr in arrays]
        if not all(np.isfinite(w).sum() >= max(30, int(0.8 * n)) for w in windows):
            continue
        out.append((i, float(d_win[0]), windows))
    return out


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref_dist, ref_features, _ = ac.build_reference_features("fwd_z_y_x_back_z_y_minusx", reference_profile="all")
    queries = ac.build_query_features("fwd_z_y_x_back_z_y_minusx")

    methods = {
        "TotalHP_NCC": {"features": ["total_raw_hp"], "mode": "ncc"},
        "AxisCal_XY_NCC": {"features": ["axis_x_hp", "axis_y_hp"], "mode": "ncc"},
        "AxisCal_XY_TotalHP_NCC": {"features": ["axis_x_hp", "axis_y_hp", "axis_total_hp"], "mode": "ncc"},
        "AxisCal_XY_MSD": {"features": ["axis_x_hp", "axis_y_hp"], "mode": "msd"},
        "SLAC_Affine_OldXYZ_to_RefXYZ": {
            "features": ["old_x_hp", "old_y_hp", "old_z_hp"],
            "ref_features": ["axis_x_hp", "axis_y_hp", "axis_z_hp"],
            "mode": "slac",
        },
    }

    rows = []
    for window_m in WINDOWS_M:
        n = int(round(window_m / STEP_M)) + 1
        ref_cache = {}
        for method, cfg in methods.items():
            if cfg["mode"] == "slac":
                continue
            ref_cache[method] = reference_window_cache(ref_dist, ref_features, cfg["features"], n)

        for q in queries:
            for method, cfg in methods.items():
                q_wins = query_windows(q, cfg["features"], n)
                if not q_wins:
                    continue
                for _, true_start, windows in q_wins:
                    if cfg["mode"] == "slac":
                        starts, scores = slac_scores(windows, ref_dist, ref_features, cfg["ref_features"], n)
                    else:
                        starts, matrices = ref_cache[method]
                        starts, scores = ncc_or_msd_scores(windows, starts, matrices, cfg["mode"])
                    stats = candidate_rank_stats(starts, scores, true_start)
                    row = {
                        "method": method,
                        "window_m": window_m,
                        "query_segment": q.segment,
                        "query_direction": q.direction,
                        "true_start_m": true_start,
                    }
                    row.update(stats)
                    rows.append(row)

    results = pd.DataFrame(rows)
    summary = summarize(results)
    results.to_csv(OUT_DIR / "short_sequence_topk_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "short_sequence_topk_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "short_sequence_topk_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_summary(summary)
    write_notes(summary)
    return results, summary


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, window_m), g in results.groupby(["method", "window_m"]):
        valid = g.dropna(subset=["top1_abs_error_m"])
        if valid.empty:
            continue
        row = {
            "method": method,
            "window_m": float(window_m),
            "query_window_count": int(len(valid)),
            "median_top1_abs_error_m": float(valid["top1_abs_error_m"].median()),
            "mean_top1_abs_error_m": float(valid["top1_abs_error_m"].mean()),
            "rmse_top1_error_m": float(np.sqrt(np.mean(np.square(valid["top1_abs_error_m"].to_numpy(float))))),
            "p75_top1_abs_error_m": float(valid["top1_abs_error_m"].quantile(0.75)),
            "median_score_gap_10m": float(valid["score_gap_10m"].median()),
        }
        for tol in [5, 10, 25]:
            for k in [1, 3, 5]:
                col = f"top{k}_within_{tol}m"
                row[col + "_rate"] = float(valid[col].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["window_m", "median_top1_abs_error_m", "method"])


def plot_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)
    methods = list(summary["method"].drop_duplicates())
    x = np.arange(len(WINDOWS_M))
    width = 0.8 / max(1, len(methods))
    for j, method in enumerate(methods):
        part = summary[summary["method"] == method].set_index("window_m").reindex(WINDOWS_M)
        axes[0].bar(x + (j - (len(methods) - 1) / 2) * width, part["median_top1_abs_error_m"], width, label=method)
        axes[1].bar(x + (j - (len(methods) - 1) / 2) * width, part["top3_within_25m_rate"], width, label=method)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{w:g} m" for w in WINDOWS_M])
    axes[0].set_ylabel("Top-1 median abs start error / m")
    axes[0].set_title("Short-sequence global initialization error")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{w:g} m" for w in WINDOWS_M])
    axes[1].set_ylabel("Correct candidate in top-3 within 25 m")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Top-k recoverability")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "short_sequence_topk_summary.png")
    plt.close(fig)


def write_notes(summary: pd.DataFrame) -> None:
    lines = [
        "# Latest-literature-aligned short-sequence experiment",
        "",
        "Purpose: test whether recent rail magnetic-localization ideas based on short spatial signatures and local sensor calibration help on the 4.14 -> 5.13 data.",
        "",
        "Implemented methods:",
        "",
        "- TotalHP_NCC: total-field high-pass normalized correlation baseline.",
        "- AxisCal_XY_NCC / MSD: manually axis-calibrated X/Y high-pass signatures.",
        "- AxisCal_XY_TotalHP_NCC: axis-calibrated X/Y plus total-field high-pass.",
        "- SLAC_Affine_OldXYZ_to_RefXYZ: for each candidate position, solve a local affine transform from query XYZ high-pass to reference XYZ high-pass and score by residual RMSE. This is a lightweight test inspired by snapshot localization with uncalibrated magnetometers; it is not yet a full SLAC implementation.",
        "",
        "Key summary:",
        "",
    ]
    if not summary.empty:
        cols = [
            "method",
            "window_m",
            "query_window_count",
            "median_top1_abs_error_m",
            "top1_within_25m_rate",
            "top3_within_25m_rate",
            "median_score_gap_10m",
        ]
        lines.append(summary[cols].to_markdown(index=False, floatfmt=".3f"))
    lines.extend(
        [
            "",
            "Interpretation rules:",
            "",
            "- High top-3 recoverability but poor top-1 means the magnetic sequence contains the right place but still has repeated false peaks; a dynamic model or confidence gate is necessary.",
            "- If SLAC improves top-k but not top-1, local axis/scale adaptation is useful as a likelihood term, but cannot replace temporal inference.",
            "- These numbers use SPAN-derived true distance only for evaluation and spatial window extraction. A deployable no-wheel method must estimate traveled distance from IMU/INS speed or treat sequence length uncertainty explicitly.",
        ]
    )
    (OUT_DIR / "latest_literature_aligned_experiment_notes.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    _, summary_df = run()
    print(summary_df.to_string(index=False))
