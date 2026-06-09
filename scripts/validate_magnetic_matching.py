from __future__ import annotations

import json
import math
import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROC_DIR = Path(r"C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\data_proc")
STEP_M = 0.5
WINDOW_LENGTHS_M = [20.0, 50.0, 100.0, 150.0]
QUERY_STRIDE_M = 5.0
SMOOTH_WINDOW_M = 5.0


def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-9:
        return x * np.nan
    return (x - mu) / sd


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        return math.nan
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < max(8, len(a) * 0.8):
        return math.nan
    return float(np.nanmean(zscore(a[mask]) * zscore(b[mask])))


def rolling_median(values: pd.Series) -> pd.Series:
    points = max(3, int(round(SMOOTH_WINDOW_M / STEP_M)) | 1)
    return values.rolling(points, center=True, min_periods=max(3, points // 3)).median()


def segment_total_columns(df: pd.DataFrame) -> list[str]:
    pattern = re.compile(r"^BMAW.*_seg\d+_mag_total$")
    return [c for c in df.columns if pattern.match(c)]


def valid_reference_windows(ref: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray]:
    values = ref["mag_total_smooth_nT"].to_numpy(float)
    pass_count = ref["pass_count"].to_numpy(float)
    starts = []
    windows = []
    for i in range(0, len(ref) - n + 1):
        v = values[i : i + n]
        pc = pass_count[i : i + n]
        if np.isfinite(v).all() and np.all(pc >= 1):
            starts.append(float(ref["distance_m"].iloc[i]))
            windows.append(zscore(v))
    if not windows:
        return np.array([]), np.empty((0, n))
    return np.array(starts, dtype=float), np.vstack(windows)


def match_query(query: np.ndarray, ref_starts: np.ndarray, ref_matrix: np.ndarray) -> dict:
    q = zscore(query)
    if len(ref_starts) == 0 or not np.isfinite(q).all():
        return {"pred_start_m": math.nan, "best_score": math.nan, "second_score": math.nan}
    scores = ref_matrix @ q / len(q)
    if not np.isfinite(scores).any():
        return {"pred_start_m": math.nan, "best_score": math.nan, "second_score": math.nan}
    best_idx = int(np.nanargmax(scores))
    best_score = float(scores[best_idx])
    best_start = float(ref_starts[best_idx])
    second_score = math.nan
    distinct = np.abs(ref_starts - best_start) >= 10.0
    if distinct.any() and np.isfinite(scores[distinct]).any():
        second_score = float(np.nanmax(scores[distinct]))
    return {
        "pred_start_m": float(best_start),
        "best_score": float(best_score),
        "second_score": float(second_score),
    }


def run_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = pd.read_csv(PROC_DIR / "magmap_4_14_fused_0p5m.csv")
    query_map = pd.read_csv(PROC_DIR / "magmap_5_13_0p5m.csv")
    ref = ref[["distance_m", "pass_count", "mag_total_smooth_nT"]].copy()
    query_dist = query_map["distance_m"].to_numpy(float)
    rows = []

    for window_m in WINDOW_LENGTHS_M:
        n = int(round(window_m / STEP_M)) + 1
        ref_starts, ref_matrix = valid_reference_windows(ref, n)
        stride_n = max(1, int(round(QUERY_STRIDE_M / STEP_M)))
        for col in segment_total_columns(query_map):
            q = rolling_median(query_map[col]).to_numpy(float)
            finite = np.isfinite(q)
            if finite.sum() < n:
                continue
            for i in range(0, len(q) - n + 1, stride_n):
                window = q[i : i + n]
                d_window = query_dist[i : i + n]
                if not np.isfinite(window).all():
                    continue
                if np.nanmax(d_window) - np.nanmin(d_window) < window_m - STEP_M:
                    continue
                result = match_query(window, ref_starts, ref_matrix)
                true_start = float(d_window[0])
                pred = result["pred_start_m"]
                err = pred - true_start if np.isfinite(pred) else math.nan
                rows.append(
                    {
                        "query_segment": col.replace("_mag_total", ""),
                        "window_m": window_m,
                        "true_start_m": true_start,
                        "true_end_m": float(d_window[-1]),
                        "pred_start_m": pred,
                        "error_m": err,
                        "abs_error_m": abs(err) if np.isfinite(err) else math.nan,
                        "best_score": result["best_score"],
                        "second_score": result["second_score"],
                        "score_gap": result["best_score"] - result["second_score"]
                        if np.isfinite(result["second_score"])
                        else math.nan,
                    }
                )
    results = pd.DataFrame(rows)
    if results.empty:
        return results, pd.DataFrame()
    summary_rows = []
    for window_m, g in results.groupby("window_m"):
        valid = g.dropna(subset=["error_m"])
        bias = float(valid["error_m"].median()) if len(valid) else math.nan
        corrected = (valid["error_m"] - bias).abs()
        summary_rows.append(
            {
                "window_m": window_m,
                "query_count": int(len(valid)),
                "median_error_bias_m": bias,
                "median_abs_error_m": float(valid["abs_error_m"].median()),
                "p75_abs_error_m": float(valid["abs_error_m"].quantile(0.75)),
                "p90_abs_error_m": float(valid["abs_error_m"].quantile(0.90)),
                "bias_corrected_median_abs_error_m": float(corrected.median()),
                "bias_corrected_p75_abs_error_m": float(corrected.quantile(0.75)),
                "median_best_score": float(valid["best_score"].median()),
                "median_score_gap": float(valid["score_gap"].median()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    results.to_csv(PROC_DIR / "matching_validation_5_13_on_4_14_total.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(PROC_DIR / "matching_validation_summary_total.csv", index=False, encoding="utf-8-sig")
    return results, summary


def plot_error_summary(results: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    paths = []
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    data = [results.loc[results["window_m"] == w, "abs_error_m"].dropna().to_numpy() for w in WINDOW_LENGTHS_M]
    ax.boxplot(data, labels=[f"{w:g} m" for w in WINDOW_LENGTHS_M], showfliers=False)
    ax.set_xlabel("待匹配磁曲线长度")
    ax.set_ylabel("绝对定位误差 / m")
    ax.set_title("5.13 total 曲线匹配到 4.14 磁图：误差分布")
    ax.grid(True, axis="y", alpha=0.25)
    path = PROC_DIR / "matching_error_box_total.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.plot(summary["window_m"], summary["median_abs_error_m"], marker="o", label="原始中位绝对误差")
    ax.plot(
        summary["window_m"],
        summary["bias_corrected_median_abs_error_m"],
        marker="o",
        label="去整体偏移后的中位绝对误差",
    )
    ax.set_xlabel("待匹配磁曲线长度 / m")
    ax.set_ylabel("误差 / m")
    ax.set_title("窗口长度对 total 磁匹配误差的影响")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = PROC_DIR / "matching_error_vs_window_total.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))
    return paths


def plot_example(results: pd.DataFrame) -> list[str]:
    ref = pd.read_csv(PROC_DIR / "magmap_4_14_fused_0p5m.csv")
    query_map = pd.read_csv(PROC_DIR / "magmap_5_13_0p5m.csv")
    paths = []
    candidates = results[(results["window_m"] == 100.0) & results["best_score"].notna()].copy()
    if candidates.empty:
        candidates = results[results["best_score"].notna()].copy()
    if candidates.empty:
        return paths
    # Use a good, confident example that is not trivially at the edge.
    candidates["rank_metric"] = candidates["abs_error_m"] + 20 * (1 - candidates["best_score"])
    row = candidates.sort_values("rank_metric").iloc[0]
    n = int(round(float(row["window_m"]) / STEP_M)) + 1
    q_col = row["query_segment"] + "_mag_total"
    query_dist = query_map["distance_m"].to_numpy(float)
    q = rolling_median(query_map[q_col]).to_numpy(float)
    i = int(round(float(row["true_start_m"]) / STEP_M))
    query = q[i : i + n]
    true_x = query_dist[i : i + n]
    pred_start = float(row["pred_start_m"])
    pred_i = int(round(pred_start / STEP_M))
    ref_y = ref["mag_total_smooth_nT"].to_numpy(float)
    pred = ref_y[pred_i : pred_i + n]
    true_ref = ref_y[i : i + n] if i + n <= len(ref_y) else np.full_like(query, np.nan)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    rel_x = np.arange(n) * STEP_M
    ax.plot(rel_x, zscore(query), lw=1.8, label=f"5.13观测：{row['query_segment']}")
    ax.plot(rel_x, zscore(pred), lw=1.8, label=f"4.14最佳匹配 @ {pred_start:.1f} m")
    if np.isfinite(true_ref).all():
        ax.plot(rel_x, zscore(true_ref), lw=1.2, alpha=0.75, label=f"4.14同距离 @ {row['true_start_m']:.1f} m")
    ax.set_xlabel("窗口内距离 / m")
    ax.set_ylabel("归一化 total")
    ax.set_title(
        f"匹配示例：窗口{row['window_m']:.0f} m，真实起点{row['true_start_m']:.1f} m，预测{pred_start:.1f} m，误差{row['error_m']:.1f} m"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = PROC_DIR / "matching_example_total.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))

    # Score curve for the same query.
    ref_starts, ref_matrix = valid_reference_windows(ref, n)
    qz = zscore(query)
    scores = ref_matrix @ qz / len(qz)
    s = pd.DataFrame({"candidate_start_m": ref_starts, "score": scores}).dropna()
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=160)
    ax.plot(s["candidate_start_m"], s["score"], lw=1.5)
    ax.axvline(row["true_start_m"], color="#2ca02c", ls="--", label="真实起点")
    ax.axvline(pred_start, color="#d62728", ls="--", label="最佳匹配")
    ax.set_xlabel("候选起点 / m")
    ax.set_ylabel("归一化互相关 NCC")
    ax.set_title("匹配得分曲线：峰值越尖锐，位置越不模糊")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = PROC_DIR / "matching_score_curve_total.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))
    return paths


def main() -> None:
    global PROC_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc-dir", type=Path, default=PROC_DIR)
    args = parser.parse_args()
    PROC_DIR = args.proc_dir
    results, summary = run_validation()
    if results.empty:
        raise SystemExit("No valid matching windows were found")
    plots = []
    plots.extend(plot_error_summary(results, summary))
    plots.extend(plot_example(results))
    report = {
        "method": "Sliding-window normalized cross-correlation. Reference: 4.14 fused total map. Query: 5.13 segment total curves.",
        "notes": [
            "Each query window is z-score normalized, so constant magnetic bias is removed.",
            "The validation is distance-domain matching; a real-time system still needs odometry/speed to convert a time segment into distance samples, or a DTW variant to absorb speed changes.",
            "The median signed error estimates a remaining map-zero or axis-origin offset between 4.14 and 5.13.",
        ],
        "summary": summary.to_dict(orient="records"),
        "outputs": {
            "window_results_csv": str(PROC_DIR / "matching_validation_5_13_on_4_14_total.csv"),
            "summary_csv": str(PROC_DIR / "matching_validation_summary_total.csv"),
            "plots": plots,
        },
    }
    (PROC_DIR / "matching_validation_report_total.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
