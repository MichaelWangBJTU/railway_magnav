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


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\imu_direction_piecewise_hmm")
AXIS_VARIANT = "fwd_z_y_x_back_z_y_minusx"
SAMPLE_PERIOD = "4s"
STEP_M = 0.5


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def find_named_file(root: Path, filename: str, must_contain: str | None = None) -> Path:
    matches = []
    for path in root.rglob(filename):
        if must_contain and must_contain not in str(path):
            continue
        matches.append(path)
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    matches.sort(key=lambda p: (len(str(p)), str(p)))
    return matches[0]


def load_track_unit() -> tuple[np.ndarray, dict[str, float]]:
    sample_path = find_named_file(Path.home() / "Desktop", "magmap_5_13_aligned_samples.csv", "data_proc_new")
    usecols = ["x_east_m", "y_north_m", "s_abs_m"]
    df = pd.read_csv(sample_path, usecols=usecols)
    df = df.dropna()
    # Fit s = a * east + b * north + c. The gradient points toward increasing
    # along-track coordinate.
    a = df[["x_east_m", "y_north_m"]].to_numpy(float)
    y = df["s_abs_m"].to_numpy(float)
    design = np.column_stack([a, np.ones(len(a))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    grad = coef[:2]
    scale = float(np.linalg.norm(grad))
    if not np.isfinite(scale) or scale < 1e-9:
        raise RuntimeError("Failed to estimate track direction")
    unit = grad / scale
    pred = design @ coef
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    meta = {
        "track_unit_east": float(unit[0]),
        "track_unit_north": float(unit[1]),
        "projection_scale": scale,
        "projection_rmse_m": rmse,
        "sample_count": int(len(df)),
    }
    return unit, meta


def load_inspvax_velocity(unit: np.ndarray) -> pd.DataFrame:
    path = find_named_file(Path.home() / "Desktop", "inspvax_parsed_summary.csv")
    df = pd.read_csv(path, usecols=["time", "north_vel_mps", "east_vel_mps"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["north_vel_mps"] = pd.to_numeric(df["north_vel_mps"], errors="coerce")
    df["east_vel_mps"] = pd.to_numeric(df["east_vel_mps"], errors="coerce")
    df = df.dropna().sort_values("time").drop_duplicates("time")
    df["track_vel_mps"] = df["east_vel_mps"].to_numpy(float) * unit[0] + df["north_vel_mps"].to_numpy(float) * unit[1]
    return df


def interp_track_velocity(times: np.ndarray, vel_df: pd.DataFrame) -> np.ndarray:
    idx = pd.to_datetime(times)
    t_q = idx.astype("int64").to_numpy(float) / 1e9
    t_src = vel_df["time"].astype("int64").to_numpy(float) / 1e9
    v_src = vel_df["track_vel_mps"].to_numpy(float)
    v = np.interp(t_q, t_src, v_src, left=np.nan, right=np.nan)
    v = (
        pd.Series(v)
        .rolling(7, center=True, min_periods=1)
        .median()
        .interpolate(limit_direction="both")
        .to_numpy(float)
    )
    return v


def detect_reversal(q: hmm.QuerySegment, vel_df: pd.DataFrame) -> dict[str, float | int]:
    v_track = interp_track_velocity(q.time, vel_df)
    nominal = 1.0 if q.direction == "forward" else -1.0
    v_progress = nominal * v_track
    finite = np.isfinite(v_progress)
    if finite.sum() < 20:
        return {
            "switch_index": -1,
            "reverse_fraction": math.nan,
            "tail_reverse_fraction": math.nan,
            "tail_reverse_distance_m": math.nan,
            "mean_progress_velocity_mps": math.nan,
        }

    threshold = -0.15
    reverse = np.where(finite, v_progress < threshold, False)
    n = len(reverse)
    min_idx = max(20, int(0.45 * n))
    min_run = max(5, int(round(24.0 / 4.0)))
    switch_idx = -1
    i = min_idx
    while i < n:
        if not reverse[i]:
            i += 1
            continue
        j = i
        while j < n and reverse[j]:
            j += 1
        if j - i >= min_run:
            switch_idx = i
            break
        i = j + 1

    if switch_idx >= 0:
        tail = v_progress[switch_idx:]
        tail_reverse_fraction = float(np.mean(tail < threshold))
        dt_s = np.diff(pd.to_datetime(q.time[switch_idx:]).astype("int64").to_numpy(float) / 1e9)
        v_mid = (tail[:-1] + tail[1:]) / 2.0
        reverse_distance = float(np.nansum(np.maximum(-v_mid, 0.0) * np.maximum(dt_s, 0.0)))
        if reverse_distance < 15.0 or tail_reverse_fraction < 0.25:
            switch_idx = -1
    else:
        tail_reverse_fraction = math.nan
        reverse_distance = math.nan

    return {
        "switch_index": int(switch_idx),
        "reverse_fraction": float(np.mean(reverse[finite])),
        "tail_reverse_fraction": float(tail_reverse_fraction) if np.isfinite(tail_reverse_fraction) else math.nan,
        "tail_reverse_distance_m": float(reverse_distance) if np.isfinite(reverse_distance) else math.nan,
        "mean_progress_velocity_mps": float(np.nanmean(v_progress)),
    }


def slice_query(q: hmm.QuerySegment, start: int, stop: int, direction: str | None = None) -> hmm.QuerySegment:
    return replace(
        q,
        direction=q.direction if direction is None else direction,
        time=q.time[start:stop],
        truth_s=q.truth_s[start:stop],
        speed_mps=q.speed_mps[start:stop],
        features={k: v[start:stop] for k, v in q.features.items()},
    )


def viterbi_with_center_prior(
    q: hmm.QuerySegment,
    ref: dict[str, np.ndarray],
    direction: str,
    start_center_m: float | None,
    vmax_mps: float = 1.2,
    start_sigma_m: float = 20.0,
) -> tuple[np.ndarray, dict[str, float]]:
    q_use = replace(q, direction=direction)
    dist = ref["distance_m"]
    n_s = len(dist)
    ll = hmm.measurement_loglikelihood(q_use, ref, ["total_raw_hp_z"], {"total_raw_hp_z": 1.0}, sigma=1.2, robust=True)
    times = pd.to_datetime(q_use.time)
    ts = times.astype("int64").to_numpy(float) / 1e9
    direction_sign = 1 if direction == "forward" else -1
    dp = np.full((len(q_use.time), n_s), -np.inf, dtype=np.float32)
    prev = np.full((len(q_use.time), n_s), -1, dtype=np.int32)
    if start_center_m is None:
        dp[0] = ll[0]
    else:
        prior = -0.5 * ((dist - start_center_m) / start_sigma_m) ** 2
        dp[0] = ll[0] + prior.astype(np.float32)
    for k in range(1, len(q_use.time)):
        dt = max(0.2, float(ts[k] - ts[k - 1]))
        max_step = max(1, int(math.ceil(vmax_mps * dt / STEP_M)))
        for j in range(n_s):
            if direction_sign > 0:
                lo = max(0, j - max_step)
                hi = j + 1
            else:
                lo = j
                hi = min(n_s, j + max_step + 1)
            cand = dp[k - 1, lo:hi]
            if cand.size == 0:
                continue
            best_rel = int(np.argmax(cand))
            best_i = lo + best_rel
            moved = abs(j - best_i) * STEP_M
            speed = moved / max(dt, 1e-3)
            dp[k, j] = cand[best_rel] - 0.025 * speed * speed + ll[k, j]
            prev[k, j] = best_i
    path_idx = np.zeros(len(q_use.time), dtype=int)
    if not np.isfinite(dp[-1]).any():
        fallback_idx = int(np.nanargmax(ll[-1]))
        return np.full(len(q_use.time), dist[fallback_idx]), {"final_score_margin": math.nan, "dp_fallback": 1.0}
    path_idx[-1] = int(np.argmax(dp[-1]))
    for k in range(len(q_use.time) - 1, 0, -1):
        path_idx[k - 1] = prev[k, path_idx[k]]
        if path_idx[k - 1] < 0:
            path_idx[k - 1] = path_idx[k]
    pred = dist[path_idx]
    meta = {
        "final_score_margin": float(np.nanmax(dp[-1]) - np.nanpercentile(dp[-1], 99)),
        "dp_fallback": 0.0,
    }
    return pred, meta


def piecewise_total_hmm(q: hmm.QuerySegment, ref: dict[str, np.ndarray], switch_index: int) -> tuple[np.ndarray, dict[str, float]]:
    if switch_index < 0 or switch_index >= len(q.time) - 20:
        pred, meta = viterbi_with_center_prior(q, ref, q.direction, None, vmax_mps=1.2)
        meta.update({"used_piecewise": 0.0, "switch_index": -1})
        return pred, meta

    q1 = slice_query(q, 0, switch_index + 1)
    pred1, meta1 = viterbi_with_center_prior(q1, ref, q.direction, None, vmax_mps=1.2)
    opposite = "backward" if q.direction == "forward" else "forward"
    q2 = slice_query(q, switch_index, len(q.time), direction=opposite)
    pred2, meta2 = viterbi_with_center_prior(q2, ref, opposite, float(pred1[-1]), vmax_mps=1.2, start_sigma_m=18.0)
    pred = np.concatenate([pred1[:-1], pred2])
    meta = {
        "used_piecewise": 1.0,
        "switch_index": int(switch_index),
        "first_final_score_margin": meta1.get("final_score_margin", math.nan),
        "second_final_score_margin": meta2.get("final_score_margin", math.nan),
        "dp_fallback": max(meta1.get("dp_fallback", 0.0), meta2.get("dp_fallback", 0.0)),
    }
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
        .sort_values(["median_abs_error_m", "mean_abs_error_m"])
    )


def plot_trajectories(traj: pd.DataFrame, path: Path) -> None:
    segments = list(dict.fromkeys(traj["segment_short"].tolist()))
    fig, axes = plt.subplots(len(segments), 1, figsize=(12.5, 2.7 * len(segments)), dpi=180, sharex=False)
    if len(segments) == 1:
        axes = [axes]
    for ax, seg in zip(axes, segments):
        g = traj[traj["segment_short"] == seg]
        for method, part in g.groupby("method"):
            if method == "truth":
                ax.plot(part["time_s"], part["s_m"], color="black", lw=1.7, label="truth")
            elif method == "PiecewiseTotal":
                ax.plot(part["time_s"], part["s_m"], color="#d95f02", lw=1.2, label="piecewise total")
            else:
                ax.plot(part["time_s"], part["s_m"], color="#1f77b4", lw=1.0, alpha=0.75, label="fixed total")
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
    unit, track_meta = load_track_unit()
    vel_df = load_inspvax_velocity(unit)
    refs, _ = arh.build_candidate_refs()
    ref = refs["forward_only"]
    queries = hmm.read_query_segments(AXIS_VARIANT, SAMPLE_PERIOD)

    rows = []
    det_rows = []
    traj_rows = []
    for q in queries:
        det = detect_reversal(q, vel_df)
        det_rows.append({"segment_label": q.label, "direction": q.direction, **det})
        pred_fixed, _ = viterbi_with_center_prior(q, ref, q.direction, None, vmax_mps=1.2)
        pred_piece, meta_piece = piecewise_total_hmm(q, ref, int(det["switch_index"]))
        for method, pred, meta in [
            ("FixedTotal_vmax1.2", pred_fixed, {}),
            ("PiecewiseTotal_IMUDirection", pred_piece, meta_piece),
        ]:
            rows.append(
                {
                    "method": method,
                    "segment_label": q.label,
                    "segment_short": q.label.replace("BMAW15230010L_", ""),
                    "direction": q.direction,
                    **evaluate(pred, q.truth_s),
                    **meta,
                }
            )
        t = (pd.to_datetime(q.time).astype("int64").to_numpy(float) - pd.to_datetime(q.time[0]).value) / 1e9
        keep = np.linspace(0, len(q.time) - 1, min(240, len(q.time))).round().astype(int)
        switch_time = float(t[int(det["switch_index"])]) if int(det["switch_index"]) >= 0 else math.nan
        for i in keep:
            traj_rows.append({"segment_label": q.label, "segment_short": q.label.replace("BMAW15230010L_", ""), "method": "truth", "time_s": t[i], "s_m": q.truth_s[i], "switch_time_s": switch_time})
            traj_rows.append({"segment_label": q.label, "segment_short": q.label.replace("BMAW15230010L_", ""), "method": "FixedTotal", "time_s": t[i], "s_m": pred_fixed[i], "switch_time_s": switch_time})
            traj_rows.append({"segment_label": q.label, "segment_short": q.label.replace("BMAW15230010L_", ""), "method": "PiecewiseTotal", "time_s": t[i], "s_m": pred_piece[i], "switch_time_s": switch_time})

    results = pd.DataFrame(rows)
    detections = pd.DataFrame(det_rows)
    summary = summarize(results)
    traj = pd.DataFrame(traj_rows)
    results.to_csv(OUT_DIR / "imu_direction_piecewise_results.csv", index=False, encoding="utf-8-sig")
    detections.to_csv(OUT_DIR / "imu_direction_reversal_detection.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "imu_direction_piecewise_summary.csv", index=False, encoding="utf-8-sig")
    traj.to_csv(OUT_DIR / "imu_direction_piecewise_trajectories.csv", index=False, encoding="utf-8-sig")
    plot_trajectories(traj, OUT_DIR / "imu_direction_piecewise_trajectories.png")
    (OUT_DIR / "imu_direction_piecewise_summary.json").write_text(
        json.dumps(
            {
                "track_meta": track_meta,
                "summary": summary.to_dict(orient="records"),
                "detections": detections.to_dict(orient="records"),
                "results": results.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Track meta:", json.dumps(track_meta, ensure_ascii=False))
    print("\nDetections:")
    print(detections.to_string(index=False))
    print("\nSummary:")
    print(summary.round(3).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
