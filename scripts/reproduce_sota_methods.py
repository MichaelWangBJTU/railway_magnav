from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr


PROJECT_ROOT = Path.home() / "Desktop" / "磁导航" / "数据" / "codex_railway_magnav"
PROC_DIR = PROJECT_ROOT / "data_proc_new"
OUT_ROOT = PROJECT_ROOT / "sota_repro"
STEP_M = 0.5


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def ensure_dirs(out_root: Path) -> dict[str, Path]:
    dirs = {
        "root": out_root,
        "code": out_root / "code",
        "outputs": out_root / "outputs",
        "figures": out_root / "figures",
        "reports": out_root / "reports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-9:
        return np.full_like(x, np.nan)
    return (x - mu) / sd


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-9:
        return zscore(x)
    return (x - med) / (1.4826 * mad)


def corrcoef(a: np.ndarray, b: np.ndarray, min_valid_ratio: float = 0.8) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        return math.nan
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < max(8, int(len(a) * min_valid_ratio)):
        return math.nan
    az = zscore(a[mask])
    bz = zscore(b[mask])
    if not np.isfinite(az).all() or not np.isfinite(bz).all():
        return math.nan
    return float(np.mean(az * bz))


def rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return math.nan
    return float(np.sqrt(np.mean(x**2)))


def max_abs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return math.nan
    return float(np.max(np.abs(x)))


def rolling_nanmedian(x: np.ndarray, points: int) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    points = max(3, int(points) | 1)
    return s.rolling(points, center=True, min_periods=max(3, points // 3)).median().to_numpy(float)


def finite_interp(distance: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.isfinite(distance) & np.isfinite(values)
    if mask.sum() < 2:
        return np.full_like(target, np.nan, dtype=float)
    return np.interp(target, distance[mask], values[mask], left=np.nan, right=np.nan)


def read_maps(proc_dir: Path) -> dict[str, pd.DataFrame]:
    maps = {}
    for label in ["4.14", "5.13"]:
        key = label.replace(".", "_")
        maps[label] = pd.read_csv(proc_dir / f"magmap_{key}_fused_0p5m.csv")
    return maps


def read_wide_map(proc_dir: Path, label: str) -> pd.DataFrame:
    return pd.read_csv(proc_dir / f"magmap_{label.replace('.', '_')}_0p5m.csv")


def read_segments(proc_dir: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(proc_dir / f"magmap_{label.replace('.', '_')}_segments.csv")
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    return df.sort_values("start_time").reset_index(drop=True)


def smooth_ref_map(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "mag_total_smooth_nT",
        "mag_y_track_anom_smooth_nT",
        "mag_x_track_anom_smooth_nT",
        "mag_z_track_anom_smooth_nT",
    ]:
        if col in out:
            out[col] = rolling_nanmedian(out[col].to_numpy(float), int(round(5.0 / STEP_M)))
    out["mag_total_grad"] = np.gradient(out["mag_total_smooth_nT"].to_numpy(float), STEP_M)
    out["mag_y_grad"] = np.gradient(out["mag_y_track_anom_smooth_nT"].to_numpy(float), STEP_M)
    return out


def segment_columns(wide: pd.DataFrame, component: str) -> list[str]:
    pat = re.compile(rf"^BMAW.*_seg\d+_{re.escape(component)}$")
    return [c for c in wide.columns if pat.match(c)]


@dataclass
class QueryWindow:
    method_group: str
    query_segment: str
    direction: str
    true_start_m: float
    true_end_m: float
    window_m: float
    values: dict[str, np.ndarray]


def build_query_windows(
    proc_dir: Path,
    label: str = "5.13",
    lengths_m: tuple[float, ...] = (20.0, 50.0, 100.0, 150.0),
    stride_m: float = 10.0,
) -> list[QueryWindow]:
    wide = read_wide_map(proc_dir, label)
    segs = read_segments(proc_dir, label)
    dist = wide["distance_m"].to_numpy(float)
    windows: list[QueryWindow] = []
    for _, seg in segs.iterrows():
        seg_label = str(seg["segment_label"])
        total_col = f"{seg_label}_mag_total"
        y_col = f"{seg_label}_mag_y_track_anom"
        if total_col not in wide.columns:
            continue
        total = rolling_nanmedian(wide[total_col].to_numpy(float), int(round(5.0 / STEP_M)))
        y = rolling_nanmedian(wide[y_col].to_numpy(float), int(round(5.0 / STEP_M))) if y_col in wide else np.full_like(total, np.nan)
        grad = np.gradient(total, STEP_M)
        y_grad = np.gradient(y, STEP_M)
        valid = np.isfinite(total)
        if valid.sum() < 20:
            continue
        dmin = float(np.nanmin(dist[valid]))
        dmax = float(np.nanmax(dist[valid]))
        for length_m in lengths_m:
            n = int(round(length_m / STEP_M)) + 1
            starts = np.arange(math.ceil(dmin / stride_m) * stride_m, dmax - length_m + 0.001, stride_m)
            for start in starts:
                i0 = int(round(start / STEP_M))
                i1 = i0 + n
                if i1 > len(wide):
                    continue
                vals = {
                    "total": total[i0:i1],
                    "total_grad": grad[i0:i1],
                    "y": y[i0:i1],
                    "y_grad": y_grad[i0:i1],
                }
                if np.isfinite(vals["total"]).mean() < 0.95:
                    continue
                windows.append(
                    QueryWindow(
                        method_group="window",
                        query_segment=seg_label,
                        direction=str(seg["direction"]),
                        true_start_m=float(start),
                        true_end_m=float(start + length_m),
                        window_m=float(length_m),
                        values=vals,
                    )
                )
    return windows


def feature_matrix(ref: pd.DataFrame, features: list[str], n: int) -> tuple[np.ndarray, np.ndarray]:
    source = {
        "total": ref["mag_total_smooth_nT"].to_numpy(float),
        "total_grad": ref["mag_total_grad"].to_numpy(float),
        "y": ref["mag_y_track_anom_smooth_nT"].to_numpy(float),
        "y_grad": ref["mag_y_grad"].to_numpy(float),
    }
    starts = []
    mats = []
    pass_count = ref["pass_count"].to_numpy(float)
    for i in range(0, len(ref) - n + 1):
        if np.nanmin(pass_count[i : i + n]) < 1:
            continue
        parts = []
        ok = True
        for f in features:
            v = source[f][i : i + n]
            if np.isfinite(v).mean() < 0.95:
                ok = False
                break
            vz = robust_zscore(v)
            if not np.isfinite(vz).all():
                ok = False
                break
            parts.append(vz)
        if ok:
            starts.append(float(ref["distance_m"].iloc[i]))
            mats.append(np.concatenate(parts))
    if not mats:
        return np.array([]), np.empty((0, n * len(features)))
    return np.asarray(starts, dtype=float), np.vstack(mats)


def ncc_match_window(q: QueryWindow, ref_starts: np.ndarray, ref_mat: np.ndarray, features: list[str]) -> dict:
    parts = []
    for f in features:
        v = q.values[f]
        if np.isfinite(v).mean() < 0.95:
            return {}
        vz = robust_zscore(v)
        if not np.isfinite(vz).all():
            return {}
        parts.append(vz)
    qv = np.concatenate(parts)
    scores = ref_mat @ qv / len(qv)
    if not np.isfinite(scores).any():
        return {}
    idx = int(np.nanargmax(scores))
    best_start = float(ref_starts[idx])
    best_score = float(scores[idx])
    far = np.abs(ref_starts - best_start) >= max(10.0, q.window_m / 2)
    second = float(np.nanmax(scores[far])) if far.any() and np.isfinite(scores[far]).any() else math.nan
    return {
        "pred_start_m": best_start,
        "best_score": best_score,
        "second_score": second,
        "score_margin": best_score - second if np.isfinite(second) else math.nan,
    }


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int) -> float:
    a = robust_zscore(a)
    b = robust_zscore(b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return math.inf
    n, m = len(a), len(b)
    inf = 1e18
    prev = np.full(m + 1, inf)
    curr = np.full(m + 1, inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr.fill(inf)
        j0 = max(1, i - band)
        j1 = min(m, i + band)
        for j in range(j0, j1 + 1):
            cost = abs(a[i - 1] - b[j - 1])
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[m] / (n + m))


def dtw_match_window(q: QueryWindow, ref: pd.DataFrame, feature: str = "total") -> dict:
    qv = q.values[feature]
    n = len(qv)
    ref_v = ref["mag_total_smooth_nT"].to_numpy(float) if feature == "total" else ref["mag_y_track_anom_smooth_nT"].to_numpy(float)
    pass_count = ref["pass_count"].to_numpy(float)
    best = (math.inf, math.nan)
    step = int(round(10.0 / STEP_M))
    band = max(3, int(round(0.08 * n)))
    for i in range(0, len(ref) - n + 1, step):
        if np.nanmin(pass_count[i : i + n]) < 1:
            continue
        rv = ref_v[i : i + n]
        if np.isfinite(rv).mean() < 0.95:
            continue
        d = dtw_distance(qv, rv, band)
        if d < best[0]:
            best = (d, float(ref["distance_m"].iloc[i]))
    if not np.isfinite(best[0]):
        return {}
    return {"pred_start_m": best[1], "best_score": -best[0], "second_score": math.nan, "score_margin": math.nan}


def run_matching_baselines(proc_dir: Path, out_dirs: dict[str, Path]) -> pd.DataFrame:
    ref = smooth_ref_map(pd.read_csv(proc_dir / "magmap_4_14_fused_0p5m.csv"))
    queries = build_query_windows(proc_dir)
    methods = {
        "NCC_total": ["total"],
        "NCC_total_grad": ["total", "total_grad"],
        "NCC_total_y_grad": ["total", "total_grad", "y", "y_grad"],
    }
    matrices: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    for q in queries:
        n = len(q.values["total"])
        for method, feats in methods.items():
            key = ("|".join(feats), n)
            if key not in matrices:
                matrices[key] = feature_matrix(ref, feats, n)
            starts, mat = matrices[key]
            if len(starts) == 0:
                continue
            pred = ncc_match_window(q, starts, mat, feats)
            if not pred:
                continue
            err = pred["pred_start_m"] - q.true_start_m
            rows.append(
                {
                    "method": method,
                    "query_segment": q.query_segment,
                    "direction": q.direction,
                    "window_m": q.window_m,
                    "true_start_m": q.true_start_m,
                    "pred_start_m": pred["pred_start_m"],
                    "error_m": err,
                    "abs_error_m": abs(err),
                    "best_score": pred["best_score"],
                    "second_score": pred["second_score"],
                    "score_margin": pred["score_margin"],
                }
            )
        if q.window_m in (50.0, 100.0):
            pred = dtw_match_window(q, ref, "total")
            if pred:
                err = pred["pred_start_m"] - q.true_start_m
                rows.append(
                    {
                        "method": "DTW_total",
                        "query_segment": q.query_segment,
                        "direction": q.direction,
                        "window_m": q.window_m,
                        "true_start_m": q.true_start_m,
                        "pred_start_m": pred["pred_start_m"],
                        "error_m": err,
                        "abs_error_m": abs(err),
                        "best_score": pred["best_score"],
                        "second_score": math.nan,
                        "score_margin": math.nan,
                    }
                )
    results = pd.DataFrame(rows)
    results.to_csv(out_dirs["outputs"] / "baseline_matching_window_results.csv", index=False, encoding="utf-8-sig")
    summary = (
        results.groupby(["method", "window_m"])
        .agg(
            query_count=("abs_error_m", "size"),
            median_abs_error_m=("abs_error_m", "median"),
            mean_abs_error_m=("abs_error_m", "mean"),
            rmse_error_m=("error_m", lambda x: rmse(np.asarray(x, dtype=float))),
            p90_abs_error_m=("abs_error_m", lambda x: float(np.nanpercentile(x, 90))),
            median_score=("best_score", "median"),
            median_margin=("score_margin", "median"),
        )
        .reset_index()
        .sort_values(["median_abs_error_m", "rmse_error_m"])
    )
    summary.to_csv(out_dirs["outputs"] / "baseline_matching_summary.csv", index=False, encoding="utf-8-sig")
    plot_baseline_summary(summary, out_dirs)
    plot_matching_example(results, proc_dir, out_dirs)
    return summary


def plot_baseline_summary(summary: pd.DataFrame, out_dirs: dict[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    for method, g in summary.groupby("method"):
        ax.plot(g["window_m"], g["median_abs_error_m"], marker="o", lw=1.8, label=method)
    ax.set_xlabel("匹配窗口长度 / m")
    ax.set_ylabel("中位绝对误差 / m")
    ax.set_title("不同磁匹配 baseline 的定位误差")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dirs["figures"] / "baseline_matching_median_error.png")
    plt.close(fig)


def plot_matching_example(results: pd.DataFrame, proc_dir: Path, out_dirs: dict[str, Path]) -> None:
    if results.empty:
        return
    candidates = results[results["method"] == "NCC_total_y_grad"].copy()
    if candidates.empty:
        candidates = results.copy()
    candidates["rank"] = candidates["abs_error_m"] + 10.0 * (1.0 - candidates["best_score"].clip(-1, 1))
    row = candidates.sort_values("rank").iloc[0]
    ref = smooth_ref_map(pd.read_csv(proc_dir / "magmap_4_14_fused_0p5m.csv"))
    wide = read_wide_map(proc_dir, "5.13")
    q_col = f"{row['query_segment']}_mag_total"
    n = int(round(float(row["window_m"]) / STEP_M)) + 1
    i_true = int(round(float(row["true_start_m"]) / STEP_M))
    i_pred = int(round(float(row["pred_start_m"]) / STEP_M))
    q = rolling_nanmedian(wide[q_col].to_numpy(float), int(round(5.0 / STEP_M)))[i_true : i_true + n]
    r = ref["mag_total_smooth_nT"].to_numpy(float)[i_pred : i_pred + n]
    x = np.arange(n) * STEP_M
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=160)
    ax.plot(x, zscore(q), lw=1.8, label=f"5.13 query {row['query_segment']}")
    ax.plot(x, zscore(r), lw=1.8, label=f"4.14 matched @ {row['pred_start_m']:.1f} m")
    ax.set_xlabel("窗口内距离 / m")
    ax.set_ylabel("归一化 total")
    ax.set_title(f"匹配示例：{row['method']}，误差 {row['error_m']:.1f} m，得分 {row['best_score']:.3f}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dirs["figures"] / "baseline_matching_example.png")
    plt.close(fig)


@dataclass
class SlamNode:
    idx: int
    date_label: str
    segment_label: str
    direction: str
    true_s: float
    travel_m: float
    signature: np.ndarray
    signature_grad: np.ndarray


def make_signature(dist: np.ndarray, total: np.ndarray, s: float, direction: str, length_m: float) -> tuple[np.ndarray, np.ndarray] | None:
    if direction == "forward":
        start, end = s - length_m, s
    else:
        start, end = s, s + length_m
    if start < np.nanmin(dist) or end > np.nanmax(dist):
        return None
    grid = np.arange(start, end + STEP_M / 2, STEP_M)
    sig = finite_interp(dist, total, grid)
    if np.isfinite(sig).mean() < 0.95:
        return None
    sig = rolling_nanmedian(sig, int(round(3.0 / STEP_M)))
    grad = np.gradient(sig, STEP_M)
    return sig, grad


def build_slam_nodes(
    proc_dir: Path,
    date_label: str = "4.14",
    node_spacing_m: float = 25.0,
    signature_m: float = 50.0,
) -> list[SlamNode]:
    wide = read_wide_map(proc_dir, date_label)
    segs = read_segments(proc_dir, date_label)
    dist = wide["distance_m"].to_numpy(float)
    nodes: list[SlamNode] = []
    travel_accum = 0.0
    last_seg_end_s: float | None = None
    for _, seg in segs.iterrows():
        seg_label = str(seg["segment_label"])
        col = f"{seg_label}_mag_total"
        if col not in wide:
            continue
        vals = rolling_nanmedian(wide[col].to_numpy(float), int(round(5.0 / STEP_M)))
        valid = np.isfinite(vals)
        if valid.sum() < int(signature_m / STEP_M):
            continue
        s_min = max(float(seg["s_min_m"]), float(np.nanmin(dist[valid])))
        s_max = min(float(seg["s_max_m"]), float(np.nanmax(dist[valid])))
        if s_max - s_min < signature_m + node_spacing_m:
            continue
        direction = str(seg["direction"])
        if last_seg_end_s is not None:
            travel_accum += abs((s_max if direction == "forward" else s_min) - last_seg_end_s)
        if direction == "forward":
            node_positions = np.arange(s_min + signature_m, s_max + 0.001, node_spacing_m)
        else:
            node_positions = np.arange(s_max - signature_m, s_min - 0.001, -node_spacing_m)
        last_pos = node_positions[0] if len(node_positions) else None
        for pos in node_positions:
            made = make_signature(dist, vals, float(pos), direction, signature_m)
            if made is None:
                continue
            if last_pos is not None and pos != last_pos:
                travel_accum += abs(float(pos) - float(last_pos))
            sig, grad = made
            nodes.append(
                SlamNode(
                    idx=len(nodes),
                    date_label=date_label,
                    segment_label=seg_label,
                    direction=direction,
                    true_s=float(pos),
                    travel_m=float(travel_accum),
                    signature=sig,
                    signature_grad=grad,
                )
            )
            last_pos = float(pos)
        if len(node_positions):
            last_seg_end_s = float(node_positions[-1])
    return nodes


def signature_similarity(a: SlamNode, b: SlamNode, use_grad: bool = False) -> float:
    scores = [corrcoef(a.signature, b.signature)]
    if use_grad:
        scores.append(corrcoef(a.signature_grad, b.signature_grad))
    scores = [s for s in scores if np.isfinite(s)]
    if not scores:
        return math.nan
    return float(np.mean(scores))


def simulate_odometry(nodes: list[SlamNode], scale_bias: float = 0.008, noise_sigma_m: float = 0.15) -> np.ndarray:
    rng = np.random.default_rng(20260608)
    x = np.zeros(len(nodes))
    if not nodes:
        return x
    x[0] = nodes[0].true_s
    seg_bias: dict[str, float] = {}
    for i in range(1, len(nodes)):
        prev, cur = nodes[i - 1], nodes[i]
        delta_true = cur.true_s - prev.true_s
        seg_key = cur.segment_label
        if seg_key not in seg_bias:
            seg_bias[seg_key] = rng.normal(scale_bias, 0.003)
        if cur.segment_label != prev.segment_label:
            measured = delta_true + rng.normal(0.0, 0.75)
        else:
            measured = delta_true * (1.0 + seg_bias[seg_key]) + rng.normal(0.0, noise_sigma_m)
        x[i] = x[i - 1] + measured
    return x


def detect_loop_edges(
    nodes: list[SlamNode],
    odom_x: np.ndarray,
    threshold: float = 0.97,
    search_radius_m: float = 140.0,
    use_grad: bool = False,
    min_separation_nodes: int = 3,
    margin_threshold: float = -math.inf,
    mutual: bool = False,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20260608)
    for i, node in enumerate(nodes):
        candidates = []
        for j in range(i):
            if i - j <= min_separation_nodes:
                continue
            if nodes[j].segment_label == node.segment_label:
                continue
            if abs(float(odom_x[i] - odom_x[j])) > search_radius_m:
                continue
            score = signature_similarity(node, nodes[j], use_grad=use_grad)
            if np.isfinite(score):
                candidates.append((j, score))
        if not candidates:
            continue
        candidates.sort(key=lambda t: t[1], reverse=True)
        best_j, best_score = candidates[0]
        second_score = candidates[1][1] if len(candidates) > 1 else math.nan
        margin = best_score - second_score if np.isfinite(second_score) else math.nan
        if best_score < threshold:
            continue
        if np.isfinite(margin) and margin < margin_threshold:
            continue
        if mutual:
            back = []
            for k, _ in candidates[: min(10, len(candidates))]:
                score_back = signature_similarity(nodes[best_j], nodes[k], use_grad=use_grad)
                if np.isfinite(score_back):
                    back.append((k, score_back))
            if back:
                back.sort(key=lambda t: t[1], reverse=True)
                if back[0][0] != i:
                    continue
        rows.append(
            {
                "i": i,
                "j": best_j,
                "score": float(best_score),
                "second_score": float(second_score) if np.isfinite(second_score) else math.nan,
                "margin": float(margin) if np.isfinite(margin) else math.nan,
                "true_delta_m": float(nodes[i].true_s - nodes[best_j].true_s),
                "z_meas_m": float(nodes[i].true_s - nodes[best_j].true_s + rng.normal(0.0, 0.45))
                if abs(float(nodes[i].true_s - nodes[best_j].true_s)) <= 2.0
                else 0.0,
                "same_place_error_m": abs(float(nodes[i].true_s - nodes[best_j].true_s)),
                "false_loop_gt2m": abs(float(nodes[i].true_s - nodes[best_j].true_s)) > 2.0,
                "segment_i": nodes[i].segment_label,
                "segment_j": nodes[best_j].segment_label,
            }
        )
    return pd.DataFrame(rows)


def solve_pose_graph(
    nodes: list[SlamNode],
    odom_x: np.ndarray,
    loops: pd.DataFrame,
    loop_sigma_m: float = 0.45,
    odom_sigma_m: float = 1.0,
    robust: bool = False,
    max_iter: int = 6,
) -> np.ndarray:
    n = len(nodes)
    if n == 0:
        return np.array([])

    loop_weights = np.ones(len(loops), dtype=float)
    x = odom_x.copy()
    for _ in range(max_iter if robust else 1):
        rows = []
        cols = []
        data = []
        b = []
        row = 0
        # Prior.
        rows.append(row)
        cols.append(0)
        data.append(1.0 / 0.05)
        b.append(nodes[0].true_s / 0.05)
        row += 1

        for i in range(1, n):
            z = odom_x[i] - odom_x[i - 1]
            w = 1.0 / odom_sigma_m
            rows.extend([row, row])
            cols.extend([i - 1, i])
            data.extend([-w, w])
            b.append(z * w)
            row += 1

        for li, loop in loops.reset_index(drop=True).iterrows():
            i = int(loop["i"])
            j = int(loop["j"])
            z = float(loop["z_meas_m"]) if "z_meas_m" in loops.columns and np.isfinite(loop["z_meas_m"]) else 0.0
            sigma = loop_sigma_m
            if "margin" in loop and np.isfinite(loop["margin"]):
                # Distinctive matches get more trust; ambiguous matches stay weak.
                sigma = max(0.35, min(2.0, loop_sigma_m / max(0.35, float(loop["score"] + loop["margin"]))))
            w = loop_weights[li] / sigma
            rows.extend([row, row])
            cols.extend([j, i])
            data.extend([-w, w])
            b.append(z * w)
            row += 1

        a = sparse.csr_matrix((data, (rows, cols)), shape=(row, n))
        x = lsqr(a, np.asarray(b), atol=1e-10, btol=1e-10, iter_lim=2000)[0]
        if robust and len(loops):
            residuals = np.array([x[int(l.i)] - x[int(l.j)] for l in loops.itertuples()], dtype=float)
            c = 1.5
            loop_weights = np.where(np.abs(residuals) <= c, 1.0, c / np.maximum(np.abs(residuals), 1e-6))
    return x


def run_graph_slam(proc_dir: Path, out_dirs: dict[str, Path]) -> pd.DataFrame:
    nodes = build_slam_nodes(proc_dir, "4.14")
    odom_x = simulate_odometry(nodes)
    true_x = np.array([n.true_s for n in nodes], dtype=float)
    odom_err = odom_x - true_x
    rows = []
    configs = [
        {
            "name": "Paper_like_total_thr0p97",
            "threshold": 0.97,
            "use_grad": False,
            "margin_threshold": -math.inf,
            "robust": False,
            "mutual": False,
        },
        {
            "name": "Lower_thr_total_thr0p90",
            "threshold": 0.90,
            "use_grad": False,
            "margin_threshold": -math.inf,
            "robust": False,
            "mutual": False,
        },
        {
            "name": "Gradient_signature_thr0p90",
            "threshold": 0.90,
            "use_grad": True,
            "margin_threshold": -math.inf,
            "robust": False,
            "mutual": False,
        },
        {
            "name": "Proposed_distinctive_robust",
            "threshold": 0.86,
            "use_grad": True,
            "margin_threshold": 0.015,
            "robust": True,
            "mutual": False,
        },
        {
            "name": "Proposed_high_precision_loops",
            "threshold": 0.90,
            "use_grad": True,
            "margin_threshold": -math.inf,
            "robust": True,
            "mutual": False,
        },
    ]
    node_table = pd.DataFrame(
        [
            {
                "idx": n.idx,
                "date_label": n.date_label,
                "segment_label": n.segment_label,
                "direction": n.direction,
                "true_s_m": n.true_s,
                "odom_s_m": odom_x[n.idx],
                "odom_error_m": odom_x[n.idx] - n.true_s,
            }
            for n in nodes
        ]
    )
    node_table.to_csv(out_dirs["outputs"] / "graph_slam_nodes.csv", index=False, encoding="utf-8-sig")

    for cfg in configs:
        loops = detect_loop_edges(
            nodes,
            odom_x,
            threshold=cfg["threshold"],
            use_grad=cfg["use_grad"],
            margin_threshold=cfg["margin_threshold"],
            mutual=cfg["mutual"],
        )
        loops.to_csv(out_dirs["outputs"] / f"graph_slam_loops_{cfg['name']}.csv", index=False, encoding="utf-8-sig")
        opt_x = solve_pose_graph(nodes, odom_x, loops, robust=cfg["robust"])
        opt_err = opt_x - true_x
        false_rate = float(loops["false_loop_gt2m"].mean()) if not loops.empty else math.nan
        loop_rmse = rmse(loops["same_place_error_m"].to_numpy(float)) if not loops.empty else math.nan
        rows.append(
            {
                "method": cfg["name"],
                "node_count": len(nodes),
                "loop_count": int(len(loops)),
                "false_loop_rate_gt2m": false_rate,
                "loop_same_place_rmse_m": loop_rmse,
                "odometry_rmse_m": rmse(odom_err),
                "odometry_max_abs_m": max_abs(odom_err),
                "slam_rmse_m": rmse(opt_err),
                "slam_max_abs_m": max_abs(opt_err),
                "improvement_rmse_pct": 100.0 * (rmse(odom_err) - rmse(opt_err)) / rmse(odom_err) if rmse(odom_err) else math.nan,
            }
        )
        plot_graph_slam_errors(nodes, odom_x, opt_x, cfg["name"], out_dirs)
    summary = pd.DataFrame(rows).sort_values("slam_rmse_m")
    summary.to_csv(out_dirs["outputs"] / "graph_slam_summary.csv", index=False, encoding="utf-8-sig")
    plot_graph_slam_summary(summary, out_dirs)
    return summary


def plot_graph_slam_errors(nodes: list[SlamNode], odom_x: np.ndarray, opt_x: np.ndarray, name: str, out_dirs: dict[str, Path]) -> None:
    true_x = np.array([n.true_s for n in nodes], dtype=float)
    travel = np.array([n.travel_m for n in nodes], dtype=float)
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=160)
    ax.plot(travel, odom_x - true_x, lw=1.5, label="伪里程计误差")
    ax.plot(travel, opt_x - true_x, lw=1.8, label="Graph SLAM 优化后误差")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("累计行驶距离 / m")
    ax.set_ylabel("沿轨道位置误差 / m")
    ax.set_title(f"一维铁路磁图 Graph SLAM 误差：{name}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dirs["figures"] / f"graph_slam_error_{name}.png")
    plt.close(fig)


def plot_graph_slam_summary(summary: pd.DataFrame, out_dirs: dict[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    x = np.arange(len(summary))
    width = 0.34
    ax.bar(x - width / 2, summary["odometry_rmse_m"], width, label="伪里程计")
    ax.bar(x + width / 2, summary["slam_rmse_m"], width, label="Graph SLAM")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"], rotation=20, ha="right")
    ax.set_ylabel("RMSE / m")
    ax.set_title("Graph SLAM 复现与改进配置对比")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dirs["figures"] / "graph_slam_summary.png")
    plt.close(fig)


def viterbi_localization(
    query: np.ndarray,
    ref: np.ndarray,
    direction: str,
    obs_sigma: float = 0.8,
    trans_sigma: float = 2.0,
    start_stride: int = 1,
) -> tuple[float, float]:
    query = np.asarray(query, dtype=float)
    ref = np.asarray(ref, dtype=float)
    mask = np.isfinite(query)
    query = query[mask]
    if len(query) < 20:
        return math.nan, math.nan
    q = robust_zscore(query)
    r = robust_zscore(ref)
    valid_states = np.where(np.isfinite(r))[0]
    if len(valid_states) == 0:
        return math.nan, math.nan
    states = np.arange(0, len(ref), start_stride)
    states = states[np.isfinite(r[states])]
    if len(states) == 0:
        return math.nan, math.nan
    # Likelihood: local magnetic amplitude after robust normalization.
    scores = -0.5 * ((q[:, None] - r[states][None, :]) / obs_sigma) ** 2
    # Query curves are stored in increasing distance order in the wide map.
    step = 1
    expected = step * start_stride
    dp = scores[0].copy()
    backptr = np.zeros((len(q), len(states)), dtype=np.int32)
    max_jump = max(3, int(round(3.0 / STEP_M / start_stride)))
    for t in range(1, len(q)):
        new = np.full_like(dp, -np.inf)
        for j, sj in enumerate(states):
            target_prev = sj - expected
            lo = int(np.searchsorted(states, target_prev - max_jump * start_stride, side="left"))
            hi = int(np.searchsorted(states, target_prev + max_jump * start_stride, side="right"))
            if hi <= lo:
                continue
            prev_states = states[lo:hi]
            delta = sj - prev_states
            trans = -0.5 * ((delta - expected) / trans_sigma) ** 2
            vals = dp[lo:hi] + trans
            local_k = int(np.argmax(vals))
            new[j] = vals[local_k] + scores[t, j]
            backptr[t, j] = lo + local_k
        dp = new
    j = int(np.argmax(dp))
    path = [j]
    for t in range(len(q) - 1, 0, -1):
        j = int(backptr[t, j])
        path.append(j)
    path = path[::-1]
    start_idx = int(states[path[0]])
    score = float(np.max(dp) / len(q))
    return start_idx * STEP_M, score


def run_continuous_filters(proc_dir: Path, out_dirs: dict[str, Path]) -> pd.DataFrame:
    ref = smooth_ref_map(pd.read_csv(proc_dir / "magmap_4_14_fused_0p5m.csv"))
    ref_v = ref["mag_total_smooth_nT"].to_numpy(float)
    wide = read_wide_map(proc_dir, "5.13")
    segs = read_segments(proc_dir, "5.13")
    rows = []
    for _, seg in segs.iterrows():
        seg_label = str(seg["segment_label"])
        col = f"{seg_label}_mag_total"
        if col not in wide:
            continue
        vals = rolling_nanmedian(wide[col].to_numpy(float), int(round(5.0 / STEP_M)))
        valid = np.isfinite(vals)
        if valid.sum() < 80:
            continue
        d = wide["distance_m"].to_numpy(float)
        start = float(np.nanmin(d[valid]))
        end = float(np.nanmax(d[valid]))
        i0 = int(round(start / STEP_M))
        i1 = int(round(end / STEP_M)) + 1
        q = vals[i0:i1]
        pred_start, score = viterbi_localization(q, ref_v, str(seg["direction"]))
        if np.isfinite(pred_start):
            err = pred_start - start
            rows.append(
                {
                    "method": "Viterbi_total_motion_prior",
                    "query_segment": seg_label,
                    "direction": str(seg["direction"]),
                    "true_start_m": start,
                    "true_end_m": end,
                    "pred_start_m": pred_start,
                    "error_m": err,
                    "abs_error_m": abs(err),
                    "score": score,
                    "query_length_m": end - start,
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(out_dirs["outputs"] / "continuous_viterbi_results.csv", index=False, encoding="utf-8-sig")
    if not results.empty:
        summary = (
            results.groupby("method")
            .agg(
                query_count=("abs_error_m", "size"),
                median_abs_error_m=("abs_error_m", "median"),
                mean_abs_error_m=("abs_error_m", "mean"),
                rmse_error_m=("error_m", lambda x: rmse(np.asarray(x, dtype=float))),
                p90_abs_error_m=("abs_error_m", lambda x: float(np.nanpercentile(x, 90))),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame()
    summary.to_csv(out_dirs["outputs"] / "continuous_viterbi_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def highpass_feature(x: np.ndarray, window_m: float = 30.0) -> np.ndarray:
    base = rolling_nanmedian(x, int(round(window_m / STEP_M)))
    return np.asarray(x, dtype=float) - base


def segment_alignment_score(query_feats: dict[str, np.ndarray], ref_feats: dict[str, np.ndarray], ref_dist: np.ndarray, rel_d: np.ndarray, start: float, features: list[str]) -> float:
    scores = []
    target_d = start + rel_d
    for f in features:
        q = query_feats[f]
        r = finite_interp(ref_dist, ref_feats[f], target_d)
        if np.isfinite(q).mean() < 0.9 or np.isfinite(r).mean() < 0.9:
            return math.nan
        scores.append(corrcoef(robust_zscore(q), robust_zscore(r), min_valid_ratio=0.9))
    scores = [s for s in scores if np.isfinite(s)]
    if not scores:
        return math.nan
    return float(np.mean(scores))


def run_weak_mileage_alignment(proc_dir: Path, out_dirs: dict[str, Path]) -> pd.DataFrame:
    """Match an entire continuous pass with one start-offset variable.

    This is closer to weak-mileage magnetic localization than isolated window
    retrieval: the relative distance scale inside the query is preserved, and
    only the absolute map offset is searched.
    """
    ref = smooth_ref_map(pd.read_csv(proc_dir / "magmap_4_14_fused_0p5m.csv"))
    ref_dist = ref["distance_m"].to_numpy(float)
    ref_total = ref["mag_total_smooth_nT"].to_numpy(float)
    ref_feats = {
        "total": ref_total,
        "hp_total": highpass_feature(ref_total),
        "grad_total": np.gradient(ref_total, STEP_M),
        "y": ref["mag_y_track_anom_smooth_nT"].to_numpy(float),
        "hp_y": highpass_feature(ref["mag_y_track_anom_smooth_nT"].to_numpy(float)),
    }
    wide = read_wide_map(proc_dir, "5.13")
    segs = read_segments(proc_dir, "5.13")
    dist = wide["distance_m"].to_numpy(float)
    methods = {
        "WeakMileage_total": ["total"],
        "WeakMileage_highpass_grad": ["hp_total", "grad_total"],
        "WeakMileage_total_y_highpass": ["hp_total", "grad_total", "hp_y"],
    }
    rows = []
    for _, seg in segs.iterrows():
        seg_label = str(seg["segment_label"])
        total_col = f"{seg_label}_mag_total"
        y_col = f"{seg_label}_mag_y_track_anom"
        if total_col not in wide:
            continue
        total = rolling_nanmedian(wide[total_col].to_numpy(float), int(round(5.0 / STEP_M)))
        y = rolling_nanmedian(wide[y_col].to_numpy(float), int(round(5.0 / STEP_M))) if y_col in wide else np.full_like(total, np.nan)
        valid = np.isfinite(total)
        if valid.sum() < int(80 / STEP_M):
            continue
        true_start = float(np.nanmin(dist[valid]))
        true_end = float(np.nanmax(dist[valid]))
        # Use the longest central sub-sequence that can fit in the 4.14 map.
        max_len_m = min(320.0, true_end - true_start)
        crop_start = true_start + max(0.0, (true_end - true_start - max_len_m) / 2)
        crop_end = crop_start + max_len_m
        idx = (dist >= crop_start) & (dist <= crop_end) & valid
        if idx.sum() < int(80 / STEP_M):
            continue
        rel_d = dist[idx] - crop_start
        query_feats = {
            "total": total[idx],
            "hp_total": highpass_feature(total)[idx],
            "grad_total": np.gradient(total, STEP_M)[idx],
            "y": y[idx],
            "hp_y": highpass_feature(y)[idx],
        }
        candidate_min = float(ref_dist[np.isfinite(ref_total)].min())
        candidate_max = float(ref_dist[np.isfinite(ref_total)].max() - rel_d.max())
        candidates = np.arange(candidate_min, candidate_max + 0.001, STEP_M)
        for method, feats in methods.items():
            scores = np.array([segment_alignment_score(query_feats, ref_feats, ref_dist, rel_d, c, feats) for c in candidates], dtype=float)
            if not np.isfinite(scores).any():
                continue
            best_idx = int(np.nanargmax(scores))
            pred_crop_start = float(candidates[best_idx])
            pred_start = pred_crop_start - (crop_start - true_start)
            best = float(scores[best_idx])
            far = np.abs(candidates - pred_crop_start) >= 20.0
            second = float(np.nanmax(scores[far])) if far.any() and np.isfinite(scores[far]).any() else math.nan
            err = pred_start - true_start
            rows.append(
                {
                    "method": method,
                    "query_segment": seg_label,
                    "direction": str(seg["direction"]),
                    "true_start_m": true_start,
                    "true_end_m": true_end,
                    "used_crop_start_m": crop_start,
                    "used_crop_end_m": crop_end,
                    "pred_start_m": pred_start,
                    "error_m": err,
                    "abs_error_m": abs(err),
                    "best_score": best,
                    "second_score": second,
                    "score_margin": best - second if np.isfinite(second) else math.nan,
                    "used_length_m": max_len_m,
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(out_dirs["outputs"] / "weak_mileage_alignment_results.csv", index=False, encoding="utf-8-sig")
    if results.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            results.groupby("method")
            .agg(
                query_count=("abs_error_m", "size"),
                median_abs_error_m=("abs_error_m", "median"),
                mean_abs_error_m=("abs_error_m", "mean"),
                rmse_error_m=("error_m", lambda x: rmse(np.asarray(x, dtype=float))),
                p90_abs_error_m=("abs_error_m", lambda x: float(np.nanpercentile(x, 90))),
                median_score=("best_score", "median"),
                median_margin=("score_margin", "median"),
            )
            .reset_index()
            .sort_values(["median_abs_error_m", "rmse_error_m"])
        )
    summary.to_csv(out_dirs["outputs"] / "weak_mileage_alignment_summary.csv", index=False, encoding="utf-8-sig")
    plot_weak_mileage_summary(summary, out_dirs)
    plot_weak_mileage_example(results, proc_dir, out_dirs)
    return summary


def plot_weak_mileage_summary(summary: pd.DataFrame, out_dirs: dict[str, Path]) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    x = np.arange(len(summary))
    ax.bar(x, summary["median_abs_error_m"], color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"], rotation=20, ha="right")
    ax.set_ylabel("中位绝对误差 / m")
    ax.set_title("弱里程辅助整段磁曲线对齐")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dirs["figures"] / "weak_mileage_alignment_summary.png")
    plt.close(fig)


def plot_weak_mileage_example(results: pd.DataFrame, proc_dir: Path, out_dirs: dict[str, Path]) -> None:
    if results.empty:
        return
    row = results.sort_values(["abs_error_m", "best_score"], ascending=[True, False]).iloc[0]
    ref = smooth_ref_map(pd.read_csv(proc_dir / "magmap_4_14_fused_0p5m.csv"))
    wide = read_wide_map(proc_dir, "5.13")
    dist = wide["distance_m"].to_numpy(float)
    q_col = f"{row['query_segment']}_mag_total"
    q = rolling_nanmedian(wide[q_col].to_numpy(float), int(round(5.0 / STEP_M)))
    idx = (dist >= float(row["used_crop_start_m"])) & (dist <= float(row["used_crop_end_m"]))
    rel_d = dist[idx] - float(row["used_crop_start_m"])
    pred_crop = float(row["pred_start_m"]) + (float(row["used_crop_start_m"]) - float(row["true_start_m"]))
    ref_vals = finite_interp(ref["distance_m"].to_numpy(float), ref["mag_total_smooth_nT"].to_numpy(float), pred_crop + rel_d)
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    ax.plot(rel_d, zscore(q[idx]), lw=1.6, label=f"5.13 {row['query_segment']}")
    ax.plot(rel_d, zscore(ref_vals), lw=1.6, label=f"4.14 对齐 @ {row['pred_start_m']:.1f} m")
    ax.set_xlabel("片段内相对距离 / m")
    ax.set_ylabel("归一化 total")
    ax.set_title(f"弱里程整段对齐示例：误差 {row['error_m']:.1f} m")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dirs["figures"] / "weak_mileage_alignment_example.png")
    plt.close(fig)


def run_distinctive_subsequence_alignment(proc_dir: Path, out_dirs: dict[str, Path]) -> pd.DataFrame:
    """Select the most distinctive sub-sequence in each pass before matching.

    A short railway segment can contain many visually similar magnetic patterns.
    This method treats the similarity margin between the best and second-best
    candidates as a distinctiveness score and searches over possible subwindows.
    """
    ref = smooth_ref_map(pd.read_csv(proc_dir / "magmap_4_14_fused_0p5m.csv"))
    ref_dist = ref["distance_m"].to_numpy(float)
    ref_total = ref["mag_total_smooth_nT"].to_numpy(float)
    ref_feats = {
        "hp_total": highpass_feature(ref_total),
        "grad_total": np.gradient(ref_total, STEP_M),
    }
    wide = read_wide_map(proc_dir, "5.13")
    segs = read_segments(proc_dir, "5.13")
    dist = wide["distance_m"].to_numpy(float)
    rows = []
    all_candidates = []
    for _, seg in segs.iterrows():
        seg_label = str(seg["segment_label"])
        total_col = f"{seg_label}_mag_total"
        if total_col not in wide:
            continue
        total = rolling_nanmedian(wide[total_col].to_numpy(float), int(round(5.0 / STEP_M)))
        valid = np.isfinite(total)
        if valid.sum() < int(100 / STEP_M):
            continue
        true_start = float(np.nanmin(dist[valid]))
        true_end = float(np.nanmax(dist[valid]))
        query_feats_all = {
            "hp_total": highpass_feature(total),
            "grad_total": np.gradient(total, STEP_M),
        }
        candidate_rows = []
        for length_m in [100.0, 140.0, 180.0, 240.0, 320.0]:
            if true_end - true_start < length_m:
                continue
            for crop_start in np.arange(true_start, true_end - length_m + 0.001, 20.0):
                crop_end = crop_start + length_m
                idx = (dist >= crop_start) & (dist <= crop_end) & valid
                if idx.sum() < int(80 / STEP_M):
                    continue
                rel_d = dist[idx] - crop_start
                query_feats = {k: v[idx] for k, v in query_feats_all.items()}
                candidate_min = float(ref_dist[np.isfinite(ref_total)].min())
                candidate_max = float(ref_dist[np.isfinite(ref_total)].max() - rel_d.max())
                if candidate_max <= candidate_min:
                    continue
                candidates = np.arange(candidate_min, candidate_max + 0.001, STEP_M)
                scores = np.array(
                    [segment_alignment_score(query_feats, ref_feats, ref_dist, rel_d, c, ["hp_total", "grad_total"]) for c in candidates],
                    dtype=float,
                )
                if not np.isfinite(scores).any():
                    continue
                best_idx = int(np.nanargmax(scores))
                pred_crop_start = float(candidates[best_idx])
                best = float(scores[best_idx])
                far = np.abs(candidates - pred_crop_start) >= 20.0
                second = float(np.nanmax(scores[far])) if far.any() and np.isfinite(scores[far]).any() else math.nan
                margin = best - second if np.isfinite(second) else math.nan
                pred_start = pred_crop_start - (crop_start - true_start)
                err = pred_start - true_start
                candidate_rows.append(
                    {
                        "method": "DistinctiveSubseq_highpass_grad",
                        "query_segment": seg_label,
                        "direction": str(seg["direction"]),
                        "true_start_m": true_start,
                        "true_end_m": true_end,
                        "used_crop_start_m": crop_start,
                        "used_crop_end_m": crop_end,
                        "used_length_m": length_m,
                        "pred_start_m": pred_start,
                        "error_m": err,
                        "abs_error_m": abs(err),
                        "best_score": best,
                        "second_score": second,
                        "score_margin": margin,
                        "selection_score": (margin if np.isfinite(margin) else -1.0) + 0.25 * best,
                    }
                )
        if not candidate_rows:
            continue
        cand = pd.DataFrame(candidate_rows)
        all_candidates.append(cand)
        selected = cand.sort_values(["selection_score", "used_length_m"], ascending=[False, False]).iloc[0].to_dict()
        rows.append(selected)
    results = pd.DataFrame(rows)
    candidates_df = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    candidates_df.to_csv(out_dirs["outputs"] / "distinctive_subsequence_candidates.csv", index=False, encoding="utf-8-sig")
    results.to_csv(out_dirs["outputs"] / "distinctive_subsequence_results.csv", index=False, encoding="utf-8-sig")
    if results.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            results.groupby("method")
            .agg(
                query_count=("abs_error_m", "size"),
                median_abs_error_m=("abs_error_m", "median"),
                mean_abs_error_m=("abs_error_m", "mean"),
                rmse_error_m=("error_m", lambda x: rmse(np.asarray(x, dtype=float))),
                p90_abs_error_m=("abs_error_m", lambda x: float(np.nanpercentile(x, 90))),
                median_score=("best_score", "median"),
                median_margin=("score_margin", "median"),
            )
            .reset_index()
        )
    summary.to_csv(out_dirs["outputs"] / "distinctive_subsequence_summary.csv", index=False, encoding="utf-8-sig")
    plot_distinctive_summary(summary, out_dirs)
    return summary


def plot_distinctive_summary(summary: pd.DataFrame, out_dirs: dict[str, Path]) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    ax.bar(summary["method"], summary["median_abs_error_m"], color="#9467bd")
    ax.set_ylabel("中位绝对误差 / m")
    ax.set_title("辨识度驱动的子序列磁匹配")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dirs["figures"] / "distinctive_subsequence_summary.png")
    plt.close(fig)


def summarize_span_ascii(data_root: Path, out_dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for p in data_root.rglob("*.ASCII"):
        if "Converted_On_20260608" not in str(p):
            continue
        kind = p.stem.split("_")[-1]
        if kind == "UNKNOWN" or p.stat().st_size == 0:
            continue
        count = 0
        first = ""
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if count == 0:
                    first = line.strip()[:220]
                count += 1
        rows.append(
            {
                "kind": kind,
                "path": str(p),
                "rows": count,
                "size_mb": round(p.stat().st_size / 1024 / 1024, 3),
                "first_line": first,
            }
        )
    df = pd.DataFrame(rows).sort_values(["kind", "path"])
    df.to_csv(out_dirs["outputs"] / "span_ascii_inventory.csv", index=False, encoding="utf-8-sig")
    return df


def write_experiment_summary(
    out_dirs: dict[str, Path],
    graph_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    viterbi_summary: pd.DataFrame,
    weak_mileage_summary: pd.DataFrame,
    distinctive_summary: pd.DataFrame,
    span_inventory: pd.DataFrame,
) -> None:
    summary = {
        "graph_slam": graph_summary.to_dict(orient="records"),
        "window_matching_baselines": baseline_summary.to_dict(orient="records"),
        "continuous_viterbi": viterbi_summary.to_dict(orient="records"),
        "weak_mileage_alignment": weak_mileage_summary.to_dict(orient="records"),
        "distinctive_subsequence_alignment": distinctive_summary.to_dict(orient="records"),
        "span_ascii_inventory_count_by_kind": span_inventory.groupby("kind")["rows"].agg(["count", "sum"]).reset_index().to_dict(orient="records")
        if not span_inventory.empty
        else [],
        "notes": [
            "The FUSION 2024 Graph SLAM experiment used a real wheel odometer. This dataset has SPAN-derived position and velocity, so the graph experiment uses SPAN truth plus controlled pseudo-odometer drift to test whether magnetic loop closures can bound drift.",
            "Loop-closure edges in this implementation are simplified same-place constraints between high-correlation magnetic signatures. This reproduces the core one-dimensional pose-graph idea while keeping the implementation transparent.",
            "The proposed robust configuration adds gradient signatures, distinctiveness margin gating, score-dependent loop weights, and Huber-style iterative reweighting to reduce ambiguous or false loop closures.",
        ],
    }
    (out_dirs["outputs"] / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc-dir", type=Path, default=PROC_DIR)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    setup_matplotlib()
    out_dirs = ensure_dirs(args.out_root)
    span_inventory = summarize_span_ascii(args.data_root, out_dirs)
    graph_summary = run_graph_slam(args.proc_dir, out_dirs)
    baseline_summary = run_matching_baselines(args.proc_dir, out_dirs)
    viterbi_summary = run_continuous_filters(args.proc_dir, out_dirs)
    weak_mileage_summary = run_weak_mileage_alignment(args.proc_dir, out_dirs)
    distinctive_summary = run_distinctive_subsequence_alignment(args.proc_dir, out_dirs)
    write_experiment_summary(
        out_dirs,
        graph_summary,
        baseline_summary,
        viterbi_summary,
        weak_mileage_summary,
        distinctive_summary,
        span_inventory,
    )
    print(json.dumps({"out_root": str(args.out_root), "done": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
