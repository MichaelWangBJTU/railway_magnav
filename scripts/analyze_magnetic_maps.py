from __future__ import annotations

import json
import math
import re
import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path.home() / "Desktop" / "磁导航" / "数据" / "codex_railway_magnav"
PROC_DIR = PROJECT_ROOT / "data_proc_new"
GRID_STEP_M = 0.5
SMOOTH_WINDOW_M = 5.0
MIN_OVERLAP_M = 80.0

COMPONENTS = {
    "x": "mag_x_track_anom",
    "y": "mag_y_track_anom",
    "z": "mag_z_track_anom",
    "total": "mag_total",
}


def segment_columns(df: pd.DataFrame, component: str) -> list[str]:
    pat = re.compile(rf"^BMAW.*_seg\d+_{component}$")
    return [c for c in df.columns if pat.match(c)]


def robust_median(values: pd.DataFrame) -> pd.Series:
    return values.median(axis=1, skipna=True)


def mad(values: pd.DataFrame, med: pd.Series) -> pd.Series:
    return (values.sub(med, axis=0).abs()).median(axis=1, skipna=True)


def build_fused_map(date_label: str) -> pd.DataFrame:
    src = PROC_DIR / f"magmap_{date_label.replace('.', '_')}_0p5m.csv"
    df = pd.read_csv(src)
    out = pd.DataFrame(
        {
            "distance_m": df["distance_m"],
            "lat": df["map_lat"],
            "lon": df["map_lon"],
            "alt_m": df["map_alt_m"],
            "track_fit_lat": df["track_fit_lat"],
            "track_fit_lon": df["track_fit_lon"],
            "pass_count": df["map_pass_count"],
        }
    )
    smooth_points = max(3, int(round(SMOOTH_WINDOW_M / GRID_STEP_M)) | 1)
    for short, comp in COMPONENTS.items():
        cols = segment_columns(df, comp)
        vals = df[cols]
        med = robust_median(vals)
        mean = vals.mean(axis=1, skipna=True)
        std = vals.std(axis=1, skipna=True)
        out[f"{comp}_median_nT"] = med
        out[f"{comp}_mean_nT"] = mean
        out[f"{comp}_std_nT"] = std
        out[f"{comp}_mad_nT"] = mad(vals, med)
        out[f"{comp}_smooth_nT"] = med.rolling(
            smooth_points, center=True, min_periods=max(3, smooth_points // 3)
        ).median()
    dst = PROC_DIR / f"magmap_{date_label.replace('.', '_')}_fused_0p5m.csv"
    out.to_csv(dst, index=False, encoding="utf-8-sig")
    return out


def zscore(a: np.ndarray) -> np.ndarray:
    mu = np.nanmean(a)
    sd = np.nanstd(a)
    if not np.isfinite(sd) or sd == 0:
        return a * np.nan
    return (a - mu) / sd


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return math.nan
    aa = zscore(a[mask])
    bb = zscore(b[mask])
    return float(np.nanmean(aa * bb))


def unbiased_errors(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return math.nan, math.nan
    diff = a[mask] - b[mask]
    diff = diff - np.nanmedian(diff)
    return float(np.sqrt(np.nanmean(diff**2))), float(np.nanmean(np.abs(diff)))


def derivative_corr(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(np.gradient(a), np.gradient(b))


def best_lag_similarity(distance: np.ndarray, a: np.ndarray, b: np.ndarray, max_lag_m: float = 60.0) -> dict:
    best = {"lag_m": 0.0, "pearson": math.nan}
    max_steps = int(round(max_lag_m / GRID_STEP_M))
    for lag in range(-max_steps, max_steps + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[: len(aa)]
        elif lag > 0:
            aa = a[: -lag]
            bb = b[lag:]
        else:
            aa = a
            bb = b
        r = pearson(aa, bb)
        if np.isfinite(r) and (not np.isfinite(best["pearson"]) or r > best["pearson"]):
            best = {"lag_m": lag * GRID_STEP_M, "pearson": float(r)}
    return best


def cross_date_similarity(f4: pd.DataFrame, f5: pd.DataFrame) -> pd.DataFrame:
    rows = []
    common = pd.merge(
        f4,
        f5,
        on="distance_m",
        suffixes=("_4_14", "_5_13"),
        how="inner",
    )
    for short, comp in COMPONENTS.items():
        col4 = f"{comp}_smooth_nT_4_14"
        col5 = f"{comp}_smooth_nT_5_13"
        a = common[col4].to_numpy(float)
        b = common[col5].to_numpy(float)
        mask = (
            np.isfinite(a)
            & np.isfinite(b)
            & (common["pass_count_4_14"].to_numpy(float) >= 1)
            & (common["pass_count_5_13"].to_numpy(float) >= 1)
        )
        rmse, mae = unbiased_errors(a[mask], b[mask])
        lag = best_lag_similarity(common["distance_m"].to_numpy(float)[mask], a[mask], b[mask])
        rows.append(
            {
                "component": short,
                "overlap_points": int(mask.sum()),
                "overlap_m": round(float(mask.sum() * GRID_STEP_M), 1),
                "pearson_r_same_distance": pearson(a[mask], b[mask]),
                "derivative_r_same_distance": derivative_corr(a[mask], b[mask]),
                "bias_removed_rmse_nT": rmse,
                "bias_removed_mae_nT": mae,
                "best_lag_m_for_max_corr": lag["lag_m"],
                "best_lag_pearson_r": lag["pearson"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(PROC_DIR / "similarity_4_14_vs_5_13.csv", index=False, encoding="utf-8-sig")
    return out


def pairwise_segment_similarity(date_label: str) -> pd.DataFrame:
    src = PROC_DIR / f"magmap_{date_label.replace('.', '_')}_0p5m.csv"
    df = pd.read_csv(src)
    rows = []
    for short, comp in COMPONENTS.items():
        cols = segment_columns(df, comp)
        for c1, c2 in combinations(cols, 2):
            g = df[["distance_m", c1, c2]].dropna()
            overlap_m = len(g) * GRID_STEP_M
            if overlap_m < MIN_OVERLAP_M:
                continue
            a = g[c1].to_numpy(float)
            b = g[c2].to_numpy(float)
            rmse, mae = unbiased_errors(a, b)
            rows.append(
                {
                    "dataset": date_label,
                    "component": short,
                    "segment_a": c1.replace(f"_{comp}", ""),
                    "segment_b": c2.replace(f"_{comp}", ""),
                    "overlap_m": round(overlap_m, 1),
                    "pearson_r": pearson(a, b),
                    "derivative_r": derivative_corr(a, b),
                    "bias_removed_rmse_nT": rmse,
                    "bias_removed_mae_nT": mae,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(
        PROC_DIR / f"similarity_{date_label.replace('.', '_')}_segments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return out


def plot_comparisons(f4: pd.DataFrame, f5: pd.DataFrame) -> list[str]:
    out_paths = []
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"4.14": "#1f77b4", "5.13": "#d62728"}
    for short, comp in COMPONENTS.items():
        col = f"{comp}_smooth_nT"
        fig, ax = plt.subplots(figsize=(12, 4.8), dpi=160)
        if short == "total":
            y4 = f4[col].to_numpy(float)
            y5 = f5[col].to_numpy(float)
            y = pd.concat([f4[col], f5[col]], ignore_index=True).dropna()
            lo = float(y.quantile(0.01))
            hi = float(y.quantile(0.99))
            margin = max(20.0, (hi - lo) * 0.12)
            ylabel = "Total / nT"
            title = "4.14 与 5.13 磁图对比：Total"
        else:
            y4 = zscore(f4[col].to_numpy(float))
            y5 = zscore(f5[col].to_numpy(float))
            lo, hi, margin = -3.5, 3.5, 0.0
            ylabel = f"{short.upper()} anomaly / sigma"
            title = f"4.14 与 5.13 标准化磁异常对比：{short.upper()}"
        ax.plot(f4["distance_m"], y4, lw=1.7, color=colors["4.14"], label="4.14")
        ax.plot(f5["distance_m"], y5, lw=1.7, color=colors["5.13"], label="5.13")
        ax.set_xlim(0, max(float(f4["distance_m"].max()), float(f5["distance_m"].max())))
        ax.set_ylim(lo - margin, hi + margin)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("沿轨道方向距离 / m")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False)
        path = PROC_DIR / f"compare_4_14_5_13_{short}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        out_paths.append(str(path))
    return out_paths


def plot_absolute_component_panels(f4: pd.DataFrame, f5: pd.DataFrame) -> list[str]:
    out_paths = []
    colors = {"4.14": "#1f77b4", "5.13": "#d62728"}
    for short, comp in COMPONENTS.items():
        if short == "total":
            continue
        col = f"{comp}_smooth_nT"
        fig, axes = plt.subplots(2, 1, figsize=(12, 6.2), dpi=160, sharex=True)
        for ax, label, df in [(axes[0], "4.14", f4), (axes[1], "5.13", f5)]:
            y = df[col].to_numpy(float)
            ax.plot(df["distance_m"], y, lw=1.5, color=colors[label], label=label)
            finite = pd.Series(y).dropna()
            if not finite.empty:
                lo = float(finite.quantile(0.01))
                hi = float(finite.quantile(0.99))
                margin = max(5.0, (hi - lo) * 0.15)
                ax.set_ylim(lo - margin, hi + margin)
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, loc="upper right")
            ax.set_ylabel(f"{short.upper()} anomaly / nT")
        axes[-1].set_xlabel("沿轨道方向距离 / m")
        fig.suptitle(f"4.14 与 5.13 {short.upper()} 磁异常绝对量级检查（分面独立纵轴）")
        path = PROC_DIR / f"compare_4_14_5_13_{short}_absolute_panels.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        out_paths.append(str(path))
    return out_paths


def plot_normalized_comparisons(f4: pd.DataFrame, f5: pd.DataFrame) -> list[str]:
    out_paths = []
    colors = {"4.14": "#1f77b4", "5.13": "#d62728"}
    for short, comp in COMPONENTS.items():
        col = f"{comp}_smooth_nT"
        y4 = zscore(f4[col].to_numpy(float))
        y5 = zscore(f5[col].to_numpy(float))
        fig, ax = plt.subplots(figsize=(12, 4.8), dpi=160)
        ax.plot(f4["distance_m"], y4, lw=1.6, color=colors["4.14"], label="4.14")
        ax.plot(f5["distance_m"], y5, lw=1.6, color=colors["5.13"], label="5.13")
        ax.set_xlim(0, max(float(f4["distance_m"].max()), float(f5["distance_m"].max())))
        ax.set_ylim(-3.5, 3.5)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("沿轨道方向距离 / m")
        ax.set_ylabel("归一化磁特征 / sigma")
        ax.set_title(f"4.14 与 5.13 归一化特征对比：{short.upper() if short != 'total' else 'Total'}")
        ax.legend(frameon=False)
        path = PROC_DIR / f"compare_4_14_5_13_{short}_normalized.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        out_paths.append(str(path))
    return out_paths


def summarize_pairwise(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    rows = []
    for comp, g in df.groupby("component"):
        rows.append(
            {
                "component": comp,
                "pair_count": int(len(g)),
                "median_pearson_r": float(g["pearson_r"].median()),
                "median_derivative_r": float(g["derivative_r"].median()),
                "median_bias_removed_rmse_nT": float(g["bias_removed_rmse_nT"].median()),
                "median_overlap_m": float(g["overlap_m"].median()),
            }
        )
    return rows


def main() -> None:
    global PROC_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc-dir", type=Path, default=PROC_DIR)
    args = parser.parse_args()
    PROC_DIR = args.proc_dir
    f4 = build_fused_map("4.14")
    f5 = build_fused_map("5.13")
    cross = cross_date_similarity(f4, f5)
    seg4 = pairwise_segment_similarity("4.14")
    seg5 = pairwise_segment_similarity("5.13")
    plots = plot_comparisons(f4, f5)
    absolute_panel_plots = plot_absolute_component_panels(f4, f5)
    normalized_plots = plot_normalized_comparisons(f4, f5)
    summary = {
        "method": {
            "fused_value": "per-distance robust median across interpolated passes",
            "smoothing_for_plots_and_similarity": f"{SMOOTH_WINDOW_M} m rolling median",
            "reason": "median/MAD is more robust to single-pass spikes and turn-around artifacts than a plain mean; mean/std are still retained in the CSV.",
        },
        "outputs": {
            "fused_4_14_csv": str(PROC_DIR / "magmap_4_14_fused_0p5m.csv"),
            "fused_5_13_csv": str(PROC_DIR / "magmap_5_13_fused_0p5m.csv"),
            "cross_date_similarity_csv": str(PROC_DIR / "similarity_4_14_vs_5_13.csv"),
            "segment_similarity_4_14_csv": str(PROC_DIR / "similarity_4_14_segments.csv"),
            "segment_similarity_5_13_csv": str(PROC_DIR / "similarity_5_13_segments.csv"),
            "comparison_plots": plots,
            "absolute_component_panel_plots": absolute_panel_plots,
            "normalized_comparison_plots": normalized_plots,
        },
        "cross_date_similarity": cross.to_dict(orient="records"),
        "within_4_14_summary": summarize_pairwise(seg4),
        "within_5_13_summary": summarize_pairwise(seg5),
    }
    path = PROC_DIR / "magmap_similarity_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
