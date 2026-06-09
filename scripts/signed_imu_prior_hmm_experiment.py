from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anchor_reference_hmm_experiment as arh
import axis_calibrated_hmm_experiment as hmm
import imu_direction_piecewise_hmm as idir


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\signed_imu_prior_hmm_experiment")
AXIS_VARIANT = "fwd_z_y_x_back_z_y_minusx"
SAMPLE_PERIOD = "4s"
STEP_M = 0.5


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def viterbi_signed_imu(
    q: hmm.QuerySegment,
    ref: dict[str, np.ndarray],
    track_vel_mps: np.ndarray,
    velocity_scale: float,
    step_sigma_m: float,
    transition_weight: float,
    vmax_mps: float,
    sign_gate: bool,
    start_center_m: float | None = None,
    start_sigma_m: float = 20.0,
) -> tuple[np.ndarray, dict[str, float]]:
    dist = ref["distance_m"]
    n_s = len(dist)
    ll = hmm.measurement_loglikelihood(q, ref, ["total_raw_hp_z"], {"total_raw_hp_z": 1.0}, sigma=1.2, robust=True)
    times = pd.to_datetime(q.time)
    ts = times.astype("int64").to_numpy(float) / 1e9
    dp = np.full((len(q.time), n_s), -np.inf, dtype=np.float32)
    prev = np.full((len(q.time), n_s), -1, dtype=np.int32)
    if start_center_m is None:
        dp[0] = ll[0]
    else:
        prior = -0.5 * ((dist - start_center_m) / max(start_sigma_m, 1e-3)) ** 2
        dp[0] = ll[0] + prior.astype(np.float32)
    expected_steps = []
    for k in range(1, len(q.time)):
        dt = max(0.2, float(ts[k] - ts[k - 1]))
        expected = float(track_vel_mps[k]) * velocity_scale * dt
        if not np.isfinite(expected):
            expected = 0.0
        expected_steps.append(expected)
        max_step_m = max(vmax_mps * dt, abs(expected) + 3.0 * step_sigma_m)
        max_step = max(2, int(math.ceil(max_step_m / STEP_M)))
        sign_threshold_m = 0.4
        for j in range(n_s):
            lo = max(0, j - max_step)
            hi = min(n_s, j + max_step + 1)
            cand = dp[k - 1, lo:hi]
            if cand.size == 0:
                continue
            idx = np.arange(lo, hi)
            delta = (j - idx) * STEP_M
            valid = np.isfinite(cand)
            if sign_gate and abs(expected) >= sign_threshold_m:
                if expected > 0:
                    valid &= delta >= -STEP_M
                else:
                    valid &= delta <= STEP_M
            if not valid.any():
                continue
            trans_penalty = -transition_weight * ((delta - expected) / max(step_sigma_m, 1e-3)) ** 2
            # A tiny acceleration-free prior keeps the path from using the full
            # allowed window when magnetic likelihood is flat.
            trans_penalty += -0.01 * (delta / max(dt, 1e-3)) ** 2
            scores = np.where(valid, cand + trans_penalty.astype(np.float32), -np.inf)
            best_rel = int(np.argmax(scores))
            best_i = lo + best_rel
            dp[k, j] = scores[best_rel] + ll[k, j]
            prev[k, j] = best_i

    path_idx = np.zeros(len(q.time), dtype=int)
    if not np.isfinite(dp[-1]).any():
        fallback_idx = int(np.nanargmax(ll[-1]))
        return np.full(len(q.time), dist[fallback_idx]), {
            "final_score_margin": math.nan,
            "dp_fallback": 1.0,
            "mean_expected_step_m": float(np.nanmean(expected_steps)) if expected_steps else math.nan,
        }
    path_idx[-1] = int(np.argmax(dp[-1]))
    for k in range(len(q.time) - 1, 0, -1):
        path_idx[k - 1] = prev[k, path_idx[k]]
        if path_idx[k - 1] < 0:
            path_idx[k - 1] = path_idx[k]
    pred = dist[path_idx]
    meta = {
        "final_score_margin": float(np.nanmax(dp[-1]) - np.nanpercentile(dp[-1], 99)),
        "dp_fallback": 0.0,
        "mean_expected_step_m": float(np.nanmean(expected_steps)) if expected_steps else math.nan,
    }
    return pred, meta


def fixed_total(q: hmm.QuerySegment, ref: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    pred, meta = hmm.viterbi_track(
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
    return pred, meta


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
        "max_abs_error_m": float(np.max(np.abs(err))),
        "within_25m_rate": float(np.mean(np.abs(err) <= 25.0)),
        "within_50m_rate": float(np.mean(np.abs(err) <= 50.0)),
    }


def configs() -> list[dict]:
    out = [
        {"method": "FixedTotal_vmax1.2", "fixed": True},
    ]
    # Keep this as a diagnostic grid. A full sweep is slow because every
    # candidate is a full Viterbi pass over the 0.5 m rail map.
    for scale, sigma, weight in [
        (0.5, 5.0, 0.04),
        (0.7, 5.0, 0.04),
        (0.7, 5.0, 0.10),
        (0.9, 5.0, 0.04),
        (0.9, 8.0, 0.04),
        (1.1, 8.0, 0.04),
    ]:
        out.append(
            {
                "method": f"SignedIMU_s{scale:g}_sig{sigma:g}_w{weight:g}",
                "fixed": False,
                "velocity_scale": scale,
                "step_sigma_m": sigma,
                "transition_weight": weight,
                "vmax_mps": 1.6,
                "sign_gate": True,
            }
        )
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
        .sort_values(["median_abs_error_m", "mean_abs_error_m", "mean_final_error_m"])
    )


def plot_best(traj: pd.DataFrame, best_method: str, path: Path) -> None:
    plot_df = traj[traj["method"].isin(["truth", "FixedTotal_vmax1.2", best_method])].copy()
    segments = list(dict.fromkeys(plot_df["segment_short"].tolist()))
    fig, axes = plt.subplots(len(segments), 1, figsize=(12.5, 2.7 * len(segments)), dpi=180, sharex=False)
    if len(segments) == 1:
        axes = [axes]
    colors = {"truth": "black", "FixedTotal_vmax1.2": "#1f77b4", best_method: "#d95f02"}
    labels = {"truth": "truth", "FixedTotal_vmax1.2": "fixed total", best_method: "signed-IMU total"}
    for ax, seg in zip(axes, segments):
        g = plot_df[plot_df["segment_short"] == seg]
        for method, part in g.groupby("method"):
            ax.plot(part["time_s"], part["s_m"], color=colors[method], lw=1.3 if method != "truth" else 1.8, label=labels[method])
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

    rows = []
    traj_rows = []
    cfgs = configs()
    for q in queries:
        track_vel = idir.interp_track_velocity(q.time, vel_df)
        t = (pd.to_datetime(q.time).astype("int64").to_numpy(float) - pd.to_datetime(q.time[0]).value) / 1e9
        keep = np.linspace(0, len(q.time) - 1, min(220, len(q.time))).round().astype(int)
        for cfg in cfgs:
            if cfg["fixed"]:
                pred, meta = fixed_total(q, ref)
            else:
                pred, meta = viterbi_signed_imu(
                    q,
                    ref,
                    track_vel,
                    velocity_scale=cfg["velocity_scale"],
                    step_sigma_m=cfg["step_sigma_m"],
                    transition_weight=cfg["transition_weight"],
                    vmax_mps=cfg["vmax_mps"],
                    sign_gate=cfg["sign_gate"],
                )
            rows.append(
                {
                    "method": cfg["method"],
                    "segment_label": q.label,
                    "segment_short": q.label.replace("BMAW15230010L_", ""),
                    "direction": q.direction,
                    "velocity_scale": cfg.get("velocity_scale", math.nan),
                    "step_sigma_m": cfg.get("step_sigma_m", math.nan),
                    "transition_weight": cfg.get("transition_weight", math.nan),
                    **evaluate(pred, q.truth_s),
                    **meta,
                }
            )
            if cfg["method"] in {"FixedTotal_vmax1.2"}:
                for i in keep:
                    traj_rows.append({"method": "truth", "segment_label": q.label, "segment_short": q.label.replace("BMAW15230010L_", ""), "time_s": t[i], "s_m": q.truth_s[i]})
                    traj_rows.append({"method": cfg["method"], "segment_label": q.label, "segment_short": q.label.replace("BMAW15230010L_", ""), "time_s": t[i], "s_m": pred[i]})
            elif cfg["method"] == "SignedIMU_s0.7_sig5_w0.1":
                for i in keep:
                    traj_rows.append({"method": cfg["method"], "segment_label": q.label, "segment_short": q.label.replace("BMAW15230010L_", ""), "time_s": t[i], "s_m": pred[i]})

    results = pd.DataFrame(rows)
    summary = summarize(results)
    best_method = str(summary[summary["method"] != "FixedTotal_vmax1.2"].iloc[0]["method"])
    traj = pd.DataFrame(traj_rows)
    results.to_csv(OUT_DIR / "signed_imu_prior_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "signed_imu_prior_summary.csv", index=False, encoding="utf-8-sig")
    traj.to_csv(OUT_DIR / "signed_imu_prior_trajectories.csv", index=False, encoding="utf-8-sig")
    plot_best(traj, "SignedIMU_s0.7_sig5_w0.1", OUT_DIR / "signed_imu_prior_example_trajectories.png")
    (OUT_DIR / "signed_imu_prior_summary.json").write_text(
        json.dumps(
            {
                "track_meta": track_meta,
                "best_method": best_method,
                "summary": summary.to_dict(orient="records"),
                "results": results.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Track meta:", json.dumps(track_meta, ensure_ascii=False))
    print("\nBest summary:")
    print(summary.round(3).head(12).to_string(index=False))
    print(f"\nBest signed-IMU method: {best_method}")
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
