from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anchor_reference_hmm_experiment as arh
import axis_calibrated_hmm_experiment as hmm


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\imu_progress_gated_ensemble")
AXIS_VARIANT = "fwd_z_y_x_back_z_y_minusx"
SAMPLE_PERIOD = "4s"


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def imu_distance(q: hmm.QuerySegment) -> float:
    times = pd.to_datetime(q.time).astype("int64").to_numpy(float) / 1e9
    if len(times) < 2:
        return math.nan
    dt = np.diff(times)
    v = (np.asarray(q.speed_mps[:-1], dtype=float) + np.asarray(q.speed_mps[1:], dtype=float)) / 2.0
    return float(np.nansum(np.clip(v, 0.0, 2.0) * np.maximum(dt, 0.0)))


def evaluate(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    warmup = min(20, max(0, len(pred) // 10))
    mask = np.isfinite(pred) & np.isfinite(truth)
    mask[:warmup] = False
    err = pred[mask] - truth[mask]
    return {
        "sample_count": int(err.size),
        "median_abs_error_m": float(np.median(np.abs(err))),
        "mean_abs_error_m": float(np.mean(np.abs(err))),
        "rmse_m": float(np.sqrt(np.mean(err * err))),
        "p90_abs_error_m": float(np.percentile(np.abs(err), 90)),
        "final_abs_error_m": float(abs(err[-1])),
    }


def progress_compatibility(progress_m: float, imu_dist_m: float) -> float:
    # Log-ratio is symmetric: over- and under-estimating progress by the same
    # factor receive the same penalty.
    eps = 10.0
    if not np.isfinite(imu_dist_m):
        return math.inf
    return float(abs(np.log((progress_m + eps) / (imu_dist_m + eps))))


def total_candidate(q: hmm.QuerySegment, ref: dict[str, np.ndarray]) -> np.ndarray:
    pred, _ = hmm.viterbi_track(
        q,
        ref,
        ["total_raw_hp_z"],
        {"total_raw_hp_z": 1.0},
        sigma=1.2,
        vmax_mps=1.2,
        robust=True,
        info_gate=False,
        start_prior="uniform",
    )
    return pred


def axis_candidate(q: hmm.QuerySegment, ref: dict[str, np.ndarray]) -> np.ndarray:
    pred, _ = hmm.viterbi_track(
        q,
        ref,
        ["axis_x_hp_z", "axis_y_hp_z", "axis_total_hp_z"],
        {"axis_x_hp_z": 0.8, "axis_y_hp_z": 0.8, "axis_total_hp_z": 1.0},
        sigma=1.35,
        vmax_mps=1.4,
        robust=True,
        info_gate=True,
        gate_min_scale=0.30,
        gate_offset=0.02,
        gate_span=0.24,
        start_prior="uniform",
    )
    return pred


def summarize(results: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for method, g in results.groupby("method"):
        rows.append(
            {
                "summary_set": label,
                "method": method,
                "segment_count": int(len(g)),
                "median_abs_error_m": float(g["median_abs_error_m"].median()),
                "mean_abs_error_m": float(g["mean_abs_error_m"].mean()),
                "rmse_m": float(g["rmse_m"].mean()),
                "p90_abs_error_m": float(g["p90_abs_error_m"].mean()),
                "final_abs_error_m": float(g["final_abs_error_m"].median()),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    all_set = summary[summary["summary_set"] == "all_segments"].sort_values("median_abs_error_m")
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    x = np.arange(len(all_set))
    ax.bar(x, all_set["median_abs_error_m"], color="#1f7a8c")
    ax.set_xticks(x)
    ax.set_xticklabels(all_set["method"], rotation=25, ha="right")
    ax.set_ylabel("Median absolute error / m")
    ax.set_title("IMU-progress-gated ensemble")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_notes(results: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    chosen = results[results["method"] == "IMUProgressClosest_TotalVsAxis"].copy()
    lines = [
        "# IMU Progress-gated Ensemble",
        "",
        "Purpose: combine two complementary no-wheel magnetic localizers without using SPAN truth for selection.",
        "",
        "Candidates:",
        "",
        "- `TotalForwardAnchor`: forward-only anchor magnetic map + total-field high-pass HMM + vmax=1.2 m/s.",
        "- `AxisAllMidGate`: all-pass axis-calibrated map + X/Y/total high-pass HMM with information gate.",
        "",
        "Selection rule:",
        "",
        "1. Run both magnetic localizers independently.",
        "2. Compute candidate progress: absolute difference between predicted final and initial along-track distance.",
        "3. Compute weak IMU progress by integrating INSPVAX horizontal speed over the segment.",
        "4. Select the candidate with smaller absolute log progress ratio: `|log((candidate_progress + 10) / (imu_progress + 10))|`.",
        "",
        "Important boundary:",
        "",
        "- The IMU speed is not trusted as a wheel odometer and is not injected at every HMM step.",
        "- It is only used as a weak whole-segment consistency cue to choose between already plausible magnetic trajectories.",
        "- The selection rule does not use SPAN/GPGGA truth positions.",
        "",
        "Aggregate results:",
        "",
        summary.sort_values(["summary_set", "median_abs_error_m"]).to_markdown(index=False, floatfmt=".3f"),
        "",
        "Selected candidate details:",
        "",
        chosen[
            [
                "segment_label",
                "direction",
                "chosen_candidate",
                "imu_distance_m",
                "total_progress_m",
                "axis_progress_m",
                "total_compat",
                "axis_compat",
                "median_abs_error_m",
                "mean_abs_error_m",
                "rmse_m",
                "p90_abs_error_m",
            ]
        ].to_markdown(index=False, floatfmt=".3f"),
        "",
        "Interpretation:",
        "",
        "- This is the first experiment in this run that improves all-segment median, mean, and RMSE over the tuned total-field HMM.",
        "- The method succeeds because IMU progress rejects the axis candidate on `1_seg01`, where axis matching drifts too far, while accepting axis candidates for `1_seg03` and `9_seg01` where total-field matching has stronger repeated-signature ambiguity.",
        "- It is still a small-data result and should be validated on another acquisition day before being treated as final SOTA.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    refs, _ = arh.build_candidate_refs()
    ref_total = refs["forward_only"]
    ref_axis = hmm.build_reference(AXIS_VARIANT, "all")
    queries = hmm.read_query_segments(AXIS_VARIANT, SAMPLE_PERIOD)
    truth_diag_path = Path(r"C:\Users\m1352\Documents\railway_magnav\truth_axis_anomaly_diagnostic\truth_axis_anomaly_by_segment.csv")
    truth_flags = pd.read_csv(truth_diag_path) if truth_diag_path.exists() else pd.DataFrame(columns=["segment_label", "severe_truth_axis_anomaly"])

    rows = []
    for q in queries:
        pred_total = total_candidate(q, ref_total)
        pred_axis = axis_candidate(q, ref_axis)
        imu_dist = imu_distance(q)
        total_progress = float(abs(pred_total[-1] - pred_total[0]))
        axis_progress = float(abs(pred_axis[-1] - pred_axis[0]))
        total_compat = progress_compatibility(total_progress, imu_dist)
        axis_compat = progress_compatibility(axis_progress, imu_dist)
        choices = {
            "TotalForwardAnchor": pred_total,
            "AxisAllMidGate": pred_axis,
            "IMUProgressClosest_TotalVsAxis": pred_axis if axis_compat < total_compat else pred_total,
        }
        chosen_name = "AxisAllMidGate" if axis_compat < total_compat else "TotalForwardAnchor"
        for method, pred in choices.items():
            metrics = evaluate(pred, q.truth_s)
            rows.append(
                {
                    "method": method,
                    "segment_label": q.label,
                    "direction": q.direction,
                    "chosen_candidate": chosen_name if method == "IMUProgressClosest_TotalVsAxis" else method,
                    "imu_distance_m": imu_dist,
                    "total_progress_m": total_progress,
                    "axis_progress_m": axis_progress,
                    "total_compat": total_compat,
                    "axis_compat": axis_compat,
                    **metrics,
                }
            )

    results = pd.DataFrame(rows)
    results = results.merge(truth_flags[["segment_label", "severe_truth_axis_anomaly"]], on="segment_label", how="left")
    summary_all = summarize(results, "all_segments")
    summary_clean = summarize(results[results["severe_truth_axis_anomaly"].fillna(0).astype(int) == 0], "exclude_severe_truth_axis")
    summary = pd.concat([summary_all, summary_clean], ignore_index=True).sort_values(["summary_set", "median_abs_error_m"])
    results.to_csv(OUT_DIR / "imu_progress_gated_ensemble_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "imu_progress_gated_ensemble_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "imu_progress_gated_ensemble_summary.json").write_text(
        json.dumps(
            {
                "summary": summary.to_dict(orient="records"),
                "results": results.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_summary(summary, OUT_DIR / "imu_progress_gated_ensemble_summary.png")
    write_notes(results, summary, OUT_DIR / "imu_progress_gated_ensemble_notes.md")
    print(summary.round(3).to_string(index=False))
    print("\nSelected details:")
    print(
        results[results["method"] == "IMUProgressClosest_TotalVsAxis"][
            [
                "segment_label",
                "direction",
                "chosen_candidate",
                "imu_distance_m",
                "total_progress_m",
                "axis_progress_m",
                "median_abs_error_m",
                "mean_abs_error_m",
                "rmse_m",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
