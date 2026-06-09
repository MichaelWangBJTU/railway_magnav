from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import research_no_wheel_sota as sota


PROJECT_ROOT = Path.home() / "Desktop" / "\u78c1\u5bfc\u822a" / "\u6570\u636e" / "codex_railway_magnav"
PROC_DIR = PROJECT_ROOT / "data_proc_new"
OUT_ROOT = PROJECT_ROOT / "no_wheel_sota" / "intra_day_4_14"
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


def robust_corr(a: np.ndarray, b: np.ndarray, min_valid_ratio: float = 0.75) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < max(20, int(len(a) * min_valid_ratio)):
        return math.nan
    az = sota.robust_z(a[mask])
    bz = sota.robust_z(b[mask])
    az = az - np.nanmean(az)
    bz = bz - np.nanmean(bz)
    denom = math.sqrt(float(np.nanmean(az * az) * np.nanmean(bz * bz)))
    if not np.isfinite(denom) or denom < 1e-9:
        return math.nan
    return float(np.nanmean(az * bz) / denom)


def smooth_feature(values: np.ndarray, points: int = 11) -> np.ndarray:
    return sota.rolling_median(np.asarray(values, dtype=float), points)


def highpass_by_grid(values: np.ndarray, window_m: float = 40.0) -> np.ndarray:
    return np.asarray(values, dtype=float) - sota.rolling_median(values, int(round(window_m / STEP_M)))


def get_segment_labels(wide: pd.DataFrame) -> list[str]:
    labels = []
    suffix = "_mag_total"
    for c in wide.columns:
        if c.endswith(suffix):
            labels.append(c[: -len(suffix)])
    return sorted(labels)


def build_lopo_reference(wide: pd.DataFrame, exclude_label: str) -> dict[str, np.ndarray]:
    labels = [s for s in get_segment_labels(wide) if s != exclude_label]
    dist = wide["distance_m"].to_numpy(float)

    def median_feature(suffix: str) -> np.ndarray:
        cols = [f"{s}_{suffix}" for s in labels if f"{s}_{suffix}" in wide.columns]
        arr = wide[cols].to_numpy(float)
        med = np.nanmedian(arr, axis=1)
        med = pd.Series(med).interpolate(limit_direction="both").to_numpy(float)
        return smooth_feature(med, 11)

    total = median_feature("mag_total")
    y = median_feature("mag_y_track_anom")
    x = median_feature("mag_x_track_anom")
    z = median_feature("mag_z_track_anom")
    total_hp = highpass_by_grid(total, 40.0)
    y_hp = highpass_by_grid(y, 40.0)
    return {
        "distance_m": dist,
        "total": total,
        "total_hp": total_hp,
        "y_hp": y_hp,
        "total_z": sota.robust_z(total),
        "total_hp_z": sota.robust_z(total_hp),
        "y_hp_z": sota.robust_z(y_hp),
        "x_z": sota.robust_z(x),
        "z_z": sota.robust_z(z),
    }


def distance_domain_lopo(wide: pd.DataFrame, segs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dist = wide["distance_m"].to_numpy(float)
    for _, seg in segs.iterrows():
        label = str(seg["segment_label"])
        col = f"{label}_mag_total"
        if col not in wide:
            continue
        ref = build_lopo_reference(wide, label)
        q_raw = wide[col].to_numpy(float)
        mask = np.isfinite(q_raw)
        if mask.sum() < 80:
            continue
        q_total_full = pd.Series(q_raw).interpolate(limit=4, limit_direction="both").to_numpy(float)
        q_total = smooth_feature(q_total_full, 11)
        q_hp = highpass_by_grid(q_total, 40.0)
        q_dist = dist[mask]
        q_feat_total = q_total[mask]
        q_feat_hp = q_hp[mask]
        true_start = float(np.nanmin(q_dist))
        length = float(np.nanmax(q_dist) - np.nanmin(q_dist))
        rel = q_dist - true_start
        if length < 60.0:
            continue
        starts = np.arange(float(np.nanmin(dist)), float(np.nanmax(dist)) - length + 0.001, 1.0)
        scores = []
        for start in starts:
            pos = start + rel
            ref_total = np.interp(pos, ref["distance_m"], ref["total"], left=np.nan, right=np.nan)
            ref_hp = np.interp(pos, ref["distance_m"], ref["total_hp"], left=np.nan, right=np.nan)
            s1 = robust_corr(q_feat_total, ref_total)
            s2 = robust_corr(q_feat_hp, ref_hp)
            scores.append(np.nanmean([s1, s2]))
        scores = np.asarray(scores, dtype=float)
        if not np.isfinite(scores).any():
            continue
        best_idx = int(np.nanargmax(scores))
        pred_start = float(starts[best_idx])
        far = np.abs(starts - pred_start) >= 20.0
        second = float(np.nanmax(scores[far])) if far.any() and np.isfinite(scores[far]).any() else math.nan
        rows.append(
            {
                "method": "LOPO_distance_corr_total_hp",
                "segment_label": label,
                "direction": seg["direction"],
                "true_start_m": true_start,
                "pred_start_m": pred_start,
                "abs_error_m": abs(pred_start - true_start),
                "coverage_length_m": length,
                "best_score": float(scores[best_idx]),
                "score_margin": float(scores[best_idx] - second) if np.isfinite(second) else math.nan,
                "sample_count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def read_query_segments_4_14(proc_dir: Path, sample_period: str = "4s") -> list[sota.QuerySegment]:
    usecols = [
        "time",
        "mag_total",
        "mag_y_track_anom",
        "mag_x_track_anom",
        "mag_z_track_anom",
        "s_abs_m",
        "segment_label",
        "direction",
    ]
    df = pd.read_csv(proc_dir / "magmap_4_14_aligned_samples.csv", usecols=usecols)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "mag_total", "s_abs_m", "segment_label", "direction"])
    queries: list[sota.QuerySegment] = []
    hp_seconds = max(11, int(round(60.0 / pd.Timedelta(sample_period).total_seconds())) | 1)
    for label, part in df.groupby("segment_label", sort=False):
        part = part.sort_values("time").set_index("time")
        direction = str(part["direction"].iloc[0])
        res = part[["mag_total", "mag_y_track_anom", "mag_x_track_anom", "mag_z_track_anom", "s_abs_m"]].resample(sample_period).median()
        res = res.interpolate(limit=3, limit_direction="both").dropna()
        if len(res) < 60:
            continue
        duration_s = (res.index[-1] - res.index[0]).total_seconds()
        net_len = abs(float(res["s_abs_m"].iloc[-1]) - float(res["s_abs_m"].iloc[0]))
        if duration_s < 60.0 or net_len < 60.0:
            continue
        total = res["mag_total"].to_numpy(float)
        y = res["mag_y_track_anom"].to_numpy(float)
        x = res["mag_x_track_anom"].to_numpy(float)
        z = res["mag_z_track_anom"].to_numpy(float)
        total_hp = sota.highpass(total, hp_seconds)
        y_hp = sota.highpass(y, hp_seconds)
        features = {
            "total_z": sota.robust_z(total),
            "total_hp_z": sota.robust_z(total_hp),
            "y_hp_z": sota.robust_z(y_hp),
            "x_z": sota.robust_z(x),
            "z_z": sota.robust_z(z),
        }
        queries.append(
            sota.QuerySegment(
                label=str(label),
                direction=direction,
                time=res.index.to_numpy(dtype="datetime64[ns]"),
                truth_s=res["s_abs_m"].to_numpy(float),
                features=features,
            )
        )
    return queries


def hmm_lopo(proc_dir: Path, wide: pd.DataFrame, sample_period: str = "4s") -> tuple[pd.DataFrame, pd.DataFrame]:
    queries = read_query_segments_4_14(proc_dir, sample_period)
    configs = [
        ("LOPO_HMM_total", ["total_z"], {"total_z": 1.0}, False),
        ("LOPO_HMM_robust_total_hp", ["total_z", "total_hp_z"], {"total_z": 0.7, "total_hp_z": 1.0}, True),
    ]
    rows = []
    traj_rows = []
    for q in queries:
        ref = build_lopo_reference(wide, q.label)
        for method, features, weights, robust in configs:
            pred, _, meta = sota.viterbi_track(
                q,
                ref,
                features,
                weights,
                sigma=1.2,
                vmax_mps=1.4,
                robust=robust,
                info_gate=False,
            )
            metrics = sota.evaluate(pred, q.truth_s, warmup=min(10, len(pred) // 10))
            rows.append({"method": method, "segment_label": q.label, "direction": q.direction, **metrics, **meta})
            take = np.linspace(0, len(pred) - 1, min(200, len(pred))).round().astype(int)
            for i in take:
                traj_rows.append(
                    {
                        "method": method,
                        "segment_label": q.label,
                        "direction": q.direction,
                        "time": str(pd.Timestamp(q.time[i])),
                        "truth_s_m": float(q.truth_s[i]),
                        "pred_s_m": float(pred[i]),
                        "error_m": float(pred[i] - q.truth_s[i]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(traj_rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    return (
        results.groupby("method")
        .agg(
            segment_count=("segment_label", "size"),
            median_abs_error_m=("median_abs_error_m", "median"),
            mean_abs_error_m=("mean_abs_error_m", "mean"),
            rmse_m=("rmse_m", "mean") if "rmse_m" in results.columns else ("abs_error_m", "mean"),
            p90_abs_error_m=("p90_abs_error_m", "mean") if "p90_abs_error_m" in results.columns else ("abs_error_m", lambda x: float(np.percentile(x, 90))),
        )
        .reset_index()
        .sort_values("median_abs_error_m")
    )


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=180)
    x = np.arange(len(summary))
    ax.bar(x, summary["median_abs_error_m"], color="#2b6cb0")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"], rotation=15, ha="right")
    ax.set_ylabel("中位绝对误差 / m")
    ax.set_title("4.14 组内留一趟验证")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_segment_errors(results: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=180)
    pivot = results.pivot_table(index="segment_label", columns="method", values="median_abs_error_m", aggfunc="first")
    if "abs_error_m" in results.columns:
        # Distance-domain rows use abs_error_m rather than trajectory median.
        dist_rows = results[results["method"].str.contains("distance")]
        for _, r in dist_rows.iterrows():
            pivot.loc[r["segment_label"], r["method"]] = r["abs_error_m"]
    pivot = pivot.sort_index()
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("绝对误差 / m")
    ax.set_title("4.14 各趟组内留一验证误差")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(proc_dir: Path, out_root: Path, sample_period: str) -> None:
    setup_matplotlib()
    dirs = ensure_dirs(out_root)
    wide = pd.read_csv(proc_dir / "magmap_4_14_0p5m.csv")
    segs = pd.read_csv(proc_dir / "magmap_4_14_segments.csv")
    distance_results = distance_domain_lopo(wide, segs)
    hmm_results, trajectories = hmm_lopo(proc_dir, wide, sample_period)

    # Normalize distance-domain result columns to combine summaries.
    dist_for_summary = distance_results.copy()
    if not dist_for_summary.empty:
        dist_for_summary["median_abs_error_m"] = dist_for_summary["abs_error_m"]
        dist_for_summary["mean_abs_error_m"] = dist_for_summary["abs_error_m"]
        dist_for_summary["rmse_m"] = dist_for_summary["abs_error_m"]
        dist_for_summary["p90_abs_error_m"] = dist_for_summary["abs_error_m"]
    combined = pd.concat([dist_for_summary, hmm_results], ignore_index=True, sort=False)
    summary = summarize(combined)

    distance_results.to_csv(dirs["outputs"] / "intra_4_14_distance_lopo_results.csv", index=False, encoding="utf-8-sig")
    hmm_results.to_csv(dirs["outputs"] / "intra_4_14_hmm_lopo_results.csv", index=False, encoding="utf-8-sig")
    trajectories.to_csv(dirs["outputs"] / "intra_4_14_hmm_lopo_trajectories.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(dirs["outputs"] / "intra_4_14_lopo_summary.csv", index=False, encoding="utf-8-sig")
    with (dirs["outputs"] / "intra_4_14_lopo_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_period": sample_period,
                "summary": summary.to_dict(orient="records"),
                "distance_results": distance_results.to_dict(orient="records"),
                "hmm_results": hmm_results.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    plot_summary(summary, dirs["figures"] / "intra_4_14_lopo_summary.png")
    plot_segment_errors(combined, dirs["figures"] / "intra_4_14_segment_errors.png")
    print(json.dumps({"out_root": str(out_root), "done": True}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc-dir", type=Path, default=PROC_DIR)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--sample-period", default="4s")
    args = parser.parse_args()
    run(args.proc_dir, args.out_root, args.sample_period)


if __name__ == "__main__":
    main()
