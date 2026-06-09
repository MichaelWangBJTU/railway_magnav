from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anchor_reference_hmm_experiment as arh
import axis_calibrated_hmm_experiment as hmm
import imu_direction_piecewise_hmm as idir
import signed_imu_prior_hmm_experiment as simuhmm


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\imu_switch_signed_suffix_experiment")
AXIS_VARIANT = "fwd_z_y_x_back_z_y_minusx"
SAMPLE_PERIOD = "4s"


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def fixed_total(q: hmm.QuerySegment, ref: dict[str, np.ndarray]) -> np.ndarray:
    pred, _ = simuhmm.fixed_total(q, ref)
    return pred


def suffix_candidate(
    q: hmm.QuerySegment,
    ref: dict[str, np.ndarray],
    switch_index: int,
    track_vel: np.ndarray,
    scale: float,
    sigma: float,
    weight: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if switch_index < 0 or switch_index >= len(q.time) - 20:
        pred = fixed_total(q, ref)
        return pred, {"used_suffix": 0.0, "switch_index": -1}
    q1 = idir.slice_query(q, 0, switch_index + 1)
    pred1 = fixed_total(q1, ref)
    q2 = idir.slice_query(q, switch_index, len(q.time))
    pred2, meta = simuhmm.viterbi_signed_imu(
        q2,
        ref,
        track_vel[switch_index:],
        velocity_scale=scale,
        step_sigma_m=sigma,
        transition_weight=weight,
        vmax_mps=1.4,
        sign_gate=True,
        start_center_m=float(pred1[-1]),
        start_sigma_m=12.0,
    )
    pred = np.concatenate([pred1[:-1], pred2])
    meta.update({"used_suffix": 1.0, "switch_index": int(switch_index), "suffix_scale": scale, "suffix_sigma": sigma, "suffix_weight": weight})
    return pred, meta


def evaluate(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return simuhmm.evaluate(pred, truth)


def configs() -> list[dict]:
    out = [{"method": "FixedTotal_vmax1.2", "fixed": True}]
    for scale, sigma, weight in [
        (0.5, 5.0, 0.04),
        (0.7, 5.0, 0.04),
        (0.9, 5.0, 0.04),
        (0.7, 8.0, 0.04),
        (0.9, 8.0, 0.04),
        (0.7, 5.0, 0.10),
    ]:
        out.append({"method": f"SwitchSuffix_s{scale:g}_sig{sigma:g}_w{weight:g}", "fixed": False, "scale": scale, "sigma": sigma, "weight": weight})
    return out


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("method")
        .agg(
            segment_count=("segment_label", "size"),
            median_abs_error_m=("median_abs_error_m", "median"),
            mean_abs_error_m=("mean_abs_error_m", "mean"),
            rmse_m=("rmse_m", "mean"),
            median_final_error_m=("final_abs_error_m", "median"),
            mean_final_error_m=("final_abs_error_m", "mean"),
            max_final_error_m=("final_abs_error_m", "max"),
            mean_within_25m_rate=("within_25m_rate", "mean"),
            mean_within_50m_rate=("within_50m_rate", "mean"),
        )
        .reset_index()
        .sort_values(["median_final_error_m", "median_abs_error_m", "mean_abs_error_m"])
    )


def plot_best(traj: pd.DataFrame, best_method: str, path: Path) -> None:
    plot_df = traj[traj["method"].isin(["truth", "FixedTotal_vmax1.2", best_method])]
    segments = list(dict.fromkeys(plot_df["segment_short"].tolist()))
    fig, axes = plt.subplots(len(segments), 1, figsize=(12.5, 2.7 * len(segments)), dpi=180, sharex=False)
    if len(segments) == 1:
        axes = [axes]
    colors = {"truth": "black", "FixedTotal_vmax1.2": "#1f77b4", best_method: "#d95f02"}
    labels = {"truth": "truth", "FixedTotal_vmax1.2": "fixed total", best_method: "IMU-switch suffix"}
    for ax, seg in zip(axes, segments):
        g = plot_df[plot_df["segment_short"] == seg]
        for method, part in g.groupby("method"):
            ax.plot(part["time_s"], part["s_m"], color=colors[method], lw=1.35 if method != "truth" else 1.8, label=labels[method])
        sw = g["switch_time_s"].dropna()
        if len(sw):
            ax.axvline(float(sw.iloc[0]), color="#b2182b", ls="--", lw=1.0, label="IMU switch")
        ax.set_title(seg)
        ax.set_ylabel("s / m")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=7)
    axes[-1].set_xlabel("segment time / s")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unit, track_meta = idir.load_track_unit()
    vel_df = idir.load_inspvax_velocity(unit)
    refs, _ = arh.build_candidate_refs()
    ref = refs["forward_only"]
    queries = hmm.read_query_segments(AXIS_VARIANT, SAMPLE_PERIOD)
    cfgs = configs()

    rows = []
    det_rows = []
    traj_rows = []
    pred_store: dict[tuple[str, str], np.ndarray] = {}
    for q in queries:
        det = idir.detect_reversal(q, vel_df)
        switch_index = int(det["switch_index"])
        track_vel = idir.interp_track_velocity(q.time, vel_df)
        det_rows.append({"segment_label": q.label, "direction": q.direction, **det})
        t = (pd.to_datetime(q.time).astype("int64").to_numpy(float) - pd.to_datetime(q.time[0]).value) / 1e9
        keep = np.linspace(0, len(q.time) - 1, min(230, len(q.time))).round().astype(int)
        switch_time = float(t[switch_index]) if switch_index >= 0 else math.nan
        for cfg in cfgs:
            if cfg["fixed"]:
                pred = fixed_total(q, ref)
                meta = {"used_suffix": 0.0, "switch_index": -1}
            else:
                pred, meta = suffix_candidate(q, ref, switch_index, track_vel, cfg["scale"], cfg["sigma"], cfg["weight"])
            pred_store[(q.label, cfg["method"])] = pred
            rows.append(
                {
                    "method": cfg["method"],
                    "segment_label": q.label,
                    "segment_short": q.label.replace("BMAW15230010L_", ""),
                    "direction": q.direction,
                    **evaluate(pred, q.truth_s),
                    **meta,
                }
            )
        for i in keep:
            traj_rows.append({"method": "truth", "segment_label": q.label, "segment_short": q.label.replace("BMAW15230010L_", ""), "time_s": t[i], "s_m": q.truth_s[i], "switch_time_s": switch_time})
            traj_rows.append({"method": "FixedTotal_vmax1.2", "segment_label": q.label, "segment_short": q.label.replace("BMAW15230010L_", ""), "time_s": t[i], "s_m": pred_store[(q.label, "FixedTotal_vmax1.2")][i], "switch_time_s": switch_time})

    results = pd.DataFrame(rows)
    summary = summarize(results)
    best_method = str(summary[summary["method"] != "FixedTotal_vmax1.2"].iloc[0]["method"])
    for q in queries:
        t = (pd.to_datetime(q.time).astype("int64").to_numpy(float) - pd.to_datetime(q.time[0]).value) / 1e9
        keep = np.linspace(0, len(q.time) - 1, min(230, len(q.time))).round().astype(int)
        det = next(row for row in det_rows if row["segment_label"] == q.label)
        switch_time = float(t[int(det["switch_index"])]) if int(det["switch_index"]) >= 0 else math.nan
        pred = pred_store[(q.label, best_method)]
        for i in keep:
            traj_rows.append({"method": best_method, "segment_label": q.label, "segment_short": q.label.replace("BMAW15230010L_", ""), "time_s": t[i], "s_m": pred[i], "switch_time_s": switch_time})

    detections = pd.DataFrame(det_rows)
    traj = pd.DataFrame(traj_rows)
    results.to_csv(OUT_DIR / "imu_switch_signed_suffix_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "imu_switch_signed_suffix_summary.csv", index=False, encoding="utf-8-sig")
    detections.to_csv(OUT_DIR / "imu_switch_signed_suffix_detections.csv", index=False, encoding="utf-8-sig")
    traj.to_csv(OUT_DIR / "imu_switch_signed_suffix_trajectories.csv", index=False, encoding="utf-8-sig")
    plot_best(traj, best_method, OUT_DIR / "imu_switch_signed_suffix_trajectories.png")
    (OUT_DIR / "imu_switch_signed_suffix_summary.json").write_text(
        json.dumps(
            {
                "track_meta": track_meta,
                "best_method": best_method,
                "summary": summary.to_dict(orient="records"),
                "detections": detections.to_dict(orient="records"),
                "results": results.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Detections:")
    print(detections.to_string(index=False))
    print("\nSummary:")
    print(summary.round(3).to_string(index=False))
    print(f"\nBest method: {best_method}")
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
