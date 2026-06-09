from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import research_no_wheel_sota as sota
import validate_4_14_intra_day as intra


PROJECT_ROOT = Path.home() / "Desktop" / "\u78c1\u5bfc\u822a" / "\u6570\u636e" / "codex_railway_magnav"
PROC_DIR = PROJECT_ROOT / "data_proc_new"
SOTA_ROOT = PROJECT_ROOT / "no_wheel_sota"
OUT_ROOT = SOTA_ROOT / "quality_gated_map"
STEP_M = 0.5


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "outputs": root / "outputs",
        "figures": root / "figures",
        "code": root / "code",
        "reports": root / "reports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def select_quality_segments(lopo_csv: Path, max_error_m: float = 2.0, min_score: float = 0.60) -> pd.DataFrame:
    df = pd.read_csv(lopo_csv)
    df["quality_pass"] = (df["abs_error_m"] <= max_error_m) & (df["best_score"] >= min_score)
    err_score = np.exp(-((df["abs_error_m"].astype(float) / 50.0) ** 2))
    corr_score = np.clip((df["best_score"].astype(float) - 0.35) / 0.35, 0.0, 1.0)
    # Soft weights keep low-consistency segments as weak evidence, instead of
    # discarding their spatial coverage and possible cross-day representiveness.
    df["quality_weight"] = 0.15 + 0.85 * err_score * corr_score
    df["quality_reason"] = np.where(
        df["quality_pass"],
        "pass",
        "reject: " + np.where(df["abs_error_m"] > max_error_m, "large_lopo_error", "low_score"),
    )
    return df


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    vals = values[mask]
    w = weights[mask]
    order = np.argsort(vals)
    vals = vals[order]
    w = w[order]
    cdf = np.cumsum(w)
    cutoff = 0.5 * cdf[-1]
    return float(vals[np.searchsorted(cdf, cutoff)])


def median_from_segments(wide: pd.DataFrame, labels: list[str], suffix: str) -> np.ndarray:
    cols = [f"{label}_{suffix}" for label in labels if f"{label}_{suffix}" in wide.columns]
    if not cols:
        return np.full(len(wide), np.nan)
    arr = wide[cols].to_numpy(float)
    med = np.nanmedian(arr, axis=1)
    med = pd.Series(med).interpolate(limit_direction="both").to_numpy(float)
    return intra.smooth_feature(med, 11)


def weighted_from_segments(wide: pd.DataFrame, labels: list[str], weights: dict[str, float], suffix: str) -> np.ndarray:
    cols = [f"{label}_{suffix}" for label in labels if f"{label}_{suffix}" in wide.columns]
    if not cols:
        return np.full(len(wide), np.nan)
    arr = wide[cols].to_numpy(float)
    w = np.array([weights[col[: -len("_" + suffix)]] for col in cols], dtype=float)
    vals = np.array([weighted_median(arr[i], w) for i in range(arr.shape[0])], dtype=float)
    vals = pd.Series(vals).interpolate(limit_direction="both").to_numpy(float)
    return intra.smooth_feature(vals, 11)


def make_ref_dict(dist: np.ndarray, total: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict[str, np.ndarray]:
    total_hp = intra.highpass_by_grid(total, 40.0)
    y_hp = intra.highpass_by_grid(y, 40.0)
    return {
        "distance_m": dist,
        "total_z": sota.robust_z(total),
        "total_hp_z": sota.robust_z(total_hp),
        "y_hp_z": sota.robust_z(y_hp),
        "x_z": sota.robust_z(x),
        "z_z": sota.robust_z(z),
    }


def build_quality_reference(proc_dir: Path, quality_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    wide = pd.read_csv(proc_dir / "magmap_4_14_0p5m.csv")
    labels = quality_df.loc[quality_df["quality_pass"], "segment_label"].astype(str).tolist()
    dist = wide["distance_m"].to_numpy(float)
    total = median_from_segments(wide, labels, "mag_total")
    x = median_from_segments(wide, labels, "mag_x_track_anom")
    y = median_from_segments(wide, labels, "mag_y_track_anom")
    z = median_from_segments(wide, labels, "mag_z_track_anom")
    total_hp = intra.highpass_by_grid(total, 40.0)
    y_hp = intra.highpass_by_grid(y, 40.0)
    pass_count = np.zeros(len(wide), dtype=int)
    for label in labels:
        col = f"{label}_mag_total"
        if col in wide.columns:
            pass_count += np.isfinite(wide[col].to_numpy(float)).astype(int)
    ref_df = pd.DataFrame(
        {
            "distance_m": dist,
            "quality_pass_count": pass_count,
            "mag_total_quality_median_nT": total,
            "mag_total_quality_hp_nT": total_hp,
            "mag_x_track_anom_quality_median_nT": x,
            "mag_y_track_anom_quality_median_nT": y,
            "mag_y_track_anom_quality_hp_nT": y_hp,
            "mag_z_track_anom_quality_median_nT": z,
        }
    )
    ref = make_ref_dict(dist, total, x, y, z)
    return ref_df, ref


def build_soft_weighted_reference(proc_dir: Path, quality_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    wide = pd.read_csv(proc_dir / "magmap_4_14_0p5m.csv")
    labels = quality_df["segment_label"].astype(str).tolist()
    weights = dict(zip(quality_df["segment_label"].astype(str), quality_df["quality_weight"].astype(float)))
    dist = wide["distance_m"].to_numpy(float)
    total = weighted_from_segments(wide, labels, weights, "mag_total")
    x = weighted_from_segments(wide, labels, weights, "mag_x_track_anom")
    y = weighted_from_segments(wide, labels, weights, "mag_y_track_anom")
    z = weighted_from_segments(wide, labels, weights, "mag_z_track_anom")
    total_hp = intra.highpass_by_grid(total, 40.0)
    y_hp = intra.highpass_by_grid(y, 40.0)
    pass_count = np.zeros(len(wide), dtype=int)
    weight_sum = np.zeros(len(wide), dtype=float)
    for label in labels:
        col = f"{label}_mag_total"
        if col in wide.columns:
            m = np.isfinite(wide[col].to_numpy(float))
            pass_count += m.astype(int)
            weight_sum += m.astype(float) * weights[label]
    ref_df = pd.DataFrame(
        {
            "distance_m": dist,
            "weighted_source_count": pass_count,
            "quality_weight_sum": weight_sum,
            "mag_total_soft_weighted_nT": total,
            "mag_total_soft_weighted_hp_nT": total_hp,
            "mag_x_track_anom_soft_weighted_nT": x,
            "mag_y_track_anom_soft_weighted_nT": y,
            "mag_y_track_anom_soft_weighted_hp_nT": y_hp,
            "mag_z_track_anom_soft_weighted_nT": z,
        }
    )
    return ref_df, make_ref_dict(dist, total, x, y, z)


def run_cross_day(proc_dir: Path, ref: dict[str, np.ndarray], sample_period: str, method_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    queries = sota.read_query_segments(proc_dir, sample_period)
    configs = [
        {
            "method": f"{method_prefix}_SOTA2018_Viterbi_total",
            "features": ["total_z"],
            "weights": {"total_z": 1.0},
            "robust": False,
        },
        {
            "method": f"{method_prefix}_RobustTotalHP_Viterbi",
            "features": ["total_z", "total_hp_z"],
            "weights": {"total_z": 0.7, "total_hp_z": 1.0},
            "robust": True,
        },
        {
            "method": f"{method_prefix}_RobustMultiFeature_Viterbi",
            "features": ["total_z", "total_hp_z", "y_hp_z"],
            "weights": {"total_z": 0.8, "total_hp_z": 1.1, "y_hp_z": 0.9},
            "robust": True,
        },
    ]
    rows = []
    traj_rows = []
    for q in queries:
        for cfg in configs:
            pred, _, meta = sota.viterbi_track(
                q,
                ref,
                cfg["features"],
                cfg["weights"],
                sigma=1.2,
                vmax_mps=1.4,
                robust=cfg["robust"],
                info_gate=False,
            )
            metrics = sota.evaluate(pred, q.truth_s, warmup=min(10, len(pred) // 10))
            rows.append({"method": cfg["method"], "segment_label": q.label, "direction": q.direction, **metrics, **meta})
            take = np.linspace(0, len(pred) - 1, min(240, len(pred))).round().astype(int)
            for i in take:
                traj_rows.append(
                    {
                        "method": cfg["method"],
                        "segment_label": q.label,
                        "direction": q.direction,
                        "time": str(pd.Timestamp(q.time[i])),
                        "truth_s_m": float(q.truth_s[i]),
                        "pred_s_m": float(pred[i]),
                        "error_m": float(pred[i] - q.truth_s[i]),
                    }
                )
    results = pd.DataFrame(rows)
    trajectories = pd.DataFrame(traj_rows)
    summary = (
        results.groupby("method")
        .agg(
            segment_count=("segment_label", "size"),
            median_abs_error_m=("median_abs_error_m", "median"),
            mean_abs_error_m=("mean_abs_error_m", "mean"),
            rmse_m=("rmse_m", "mean"),
            p90_abs_error_m=("p90_abs_error_m", "mean"),
            final_abs_error_m=("final_abs_error_m", "median"),
        )
        .reset_index()
        .sort_values("median_abs_error_m")
    )
    return results, summary, trajectories


def compare_with_original(original_summary_csv: Path, quality_summary: pd.DataFrame, soft_summary: pd.DataFrame) -> pd.DataFrame:
    original = pd.read_csv(original_summary_csv)
    keep = ["SOTA2018_Viterbi_total", "Proposed_RobustTotalHP_Viterbi", "Proposed_RobustMultiFeature_Viterbi"]
    original = original[original["method"].isin(keep)].copy()
    original["reference_map"] = "original_4_14_fused"
    quality = quality_summary.copy()
    quality["reference_map"] = "quality_gated_4_14"
    soft = soft_summary.copy()
    soft["reference_map"] = "soft_quality_weighted_4_14"
    method_map = {
        "QualityRef_SOTA2018_Viterbi_total": "SOTA2018_Viterbi_total",
        "QualityRef_RobustTotalHP_Viterbi": "Proposed_RobustTotalHP_Viterbi",
        "QualityRef_RobustMultiFeature_Viterbi": "Proposed_RobustMultiFeature_Viterbi",
        "SoftQualityRef_SOTA2018_Viterbi_total": "SOTA2018_Viterbi_total",
        "SoftQualityRef_RobustTotalHP_Viterbi": "Proposed_RobustTotalHP_Viterbi",
        "SoftQualityRef_RobustMultiFeature_Viterbi": "Proposed_RobustMultiFeature_Viterbi",
    }
    quality["method"] = quality["method"].replace(method_map)
    soft["method"] = soft["method"].replace(method_map)
    return pd.concat([original, quality, soft], ignore_index=True, sort=False)


def plot_quality_selection(quality_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=180)
    colors = np.where(quality_df["quality_pass"], "#2f855a", "#c53030")
    ax.bar(quality_df["segment_label"], quality_df["abs_error_m"], color=colors)
    ax.axhline(2.0, color="black", ls="--", lw=1, label="2 m gate")
    ax.set_ylabel("4.14组内留一距离域误差 / m")
    ax.set_title("质量门控：由4.14组内一致性筛选参考图构建趟")
    ax.tick_params(axis="x", rotation=75)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_comparison(compare: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=180)
    methods = list(compare["method"].drop_duplicates())
    x = np.arange(len(methods))
    width = 0.36
    ref_names = ["original_4_14_fused", "quality_gated_4_14", "soft_quality_weighted_4_14"]
    offsets = [-width, 0, width]
    for offset, ref_name in zip(offsets, ref_names):
        vals = []
        for method in methods:
            row = compare[(compare["method"] == method) & (compare["reference_map"] == ref_name)]
            vals.append(float(row["median_abs_error_m"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + offset, vals, width=width, label=ref_name)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("5.13跨日中位绝对误差 / m")
    ax.set_title("原始4.14磁图 vs 质量门控4.14磁图")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(proc_dir: Path, sota_root: Path, out_root: Path, sample_period: str) -> None:
    setup_matplotlib()
    dirs = ensure_dirs(out_root)
    lopo_csv = sota_root / "intra_day_4_14" / "outputs" / "intra_4_14_distance_lopo_results.csv"
    original_summary_csv = sota_root / "outputs" / "no_wheel_sota_summary.csv"
    quality_df = select_quality_segments(lopo_csv)
    ref_df, ref = build_quality_reference(proc_dir, quality_df)
    soft_ref_df, soft_ref = build_soft_weighted_reference(proc_dir, quality_df)
    results, summary, trajectories = run_cross_day(proc_dir, ref, sample_period, "QualityRef")
    soft_results, soft_summary, soft_trajectories = run_cross_day(proc_dir, soft_ref, sample_period, "SoftQualityRef")
    compare = compare_with_original(original_summary_csv, summary, soft_summary)

    quality_df.to_csv(dirs["outputs"] / "quality_gate_4_14_segments.csv", index=False, encoding="utf-8-sig")
    ref_df.to_csv(dirs["outputs"] / "quality_gated_4_14_reference_0p5m.csv", index=False, encoding="utf-8-sig")
    soft_ref_df.to_csv(dirs["outputs"] / "soft_quality_weighted_4_14_reference_0p5m.csv", index=False, encoding="utf-8-sig")
    results.to_csv(dirs["outputs"] / "quality_ref_5_13_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(dirs["outputs"] / "quality_ref_5_13_summary.csv", index=False, encoding="utf-8-sig")
    trajectories.to_csv(dirs["outputs"] / "quality_ref_5_13_trajectories.csv", index=False, encoding="utf-8-sig")
    soft_results.to_csv(dirs["outputs"] / "soft_quality_ref_5_13_results.csv", index=False, encoding="utf-8-sig")
    soft_summary.to_csv(dirs["outputs"] / "soft_quality_ref_5_13_summary.csv", index=False, encoding="utf-8-sig")
    soft_trajectories.to_csv(dirs["outputs"] / "soft_quality_ref_5_13_trajectories.csv", index=False, encoding="utf-8-sig")
    compare.to_csv(dirs["outputs"] / "original_vs_quality_ref_summary.csv", index=False, encoding="utf-8-sig")
    with (dirs["outputs"] / "quality_gated_experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_period": sample_period,
                "quality_gate": {"max_error_m": 2.0, "min_score": 0.60},
                "quality_segments": quality_df.to_dict(orient="records"),
                "summary": summary.to_dict(orient="records"),
                "soft_summary": soft_summary.to_dict(orient="records"),
                "comparison": compare.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    plot_quality_selection(quality_df, dirs["figures"] / "quality_gate_4_14_selection.png")
    plot_comparison(compare, dirs["figures"] / "original_vs_quality_ref_summary.png")
    print(json.dumps({"out_root": str(out_root), "done": True}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc-dir", type=Path, default=PROC_DIR)
    parser.add_argument("--sota-root", type=Path, default=SOTA_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--sample-period", default="4s")
    args = parser.parse_args()
    run(args.proc_dir, args.sota_root, args.out_root, args.sample_period)


if __name__ == "__main__":
    main()
