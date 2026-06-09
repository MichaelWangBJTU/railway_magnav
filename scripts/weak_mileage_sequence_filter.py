from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anchor_reference_hmm_experiment as arh
import axis_calibrated_hmm_experiment as hmm
from verify_turnaround_and_trim import trim_stop_index


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\weak_mileage_sequence_filter")
AXIS_VARIANT = "fwd_z_y_x_back_z_y_minusx"
SAMPLE_PERIOD = "4s"
STEP_M = 0.5


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def cumulative_imu_distance(q: hmm.QuerySegment) -> np.ndarray:
    times = pd.to_datetime(q.time).astype("int64").to_numpy(float) / 1e9
    dt = np.diff(times, prepend=times[0])
    v = np.asarray(q.speed_mps, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.clip(v, 0.0, 2.0)
    return np.cumsum(v * np.maximum(dt, 0.0))


def weak_mileage_filter(
    q: hmm.QuerySegment,
    ref: dict[str, np.ndarray],
    scales: list[float],
    window_samples: int | None,
    feature: str = "total_raw_hp_z",
    sigma: float = 1.2,
) -> tuple[np.ndarray, dict[str, float]]:
    dist = ref["distance_m"]
    ll = hmm.measurement_loglikelihood(q, ref, [feature], {feature: 1.0}, sigma=sigma, robust=True)
    n_t, n_s = ll.shape
    direction_sign = 1 if q.direction == "forward" else -1
    d_imu = cumulative_imu_distance(q)
    start_idx = np.arange(n_s, dtype=int)
    score = np.zeros((len(scales), n_t, n_s), dtype=np.float32)
    valid = np.zeros((len(scales), n_t, n_s), dtype=bool)

    for si, scale in enumerate(scales):
        for k in range(n_t):
            delta_idx = int(round(direction_sign * scale * d_imu[k] / STEP_M))
            pos = start_idx + delta_idx
            ok = (pos >= 0) & (pos < n_s)
            valid[si, k, ok] = True
            row = np.full(n_s, -8.0, dtype=np.float32)
            row[ok] = ll[k, pos[ok]]
            score[si, k] = row

    if window_samples is None or window_samples >= n_t:
        summed = np.cumsum(score, axis=1)
        valid_count = np.cumsum(valid.astype(np.int16), axis=1)
        method_window = n_t
    else:
        csum = np.cumsum(score, axis=1)
        vcum = np.cumsum(valid.astype(np.int16), axis=1)
        summed = csum.copy()
        valid_count = vcum.copy()
        summed[:, window_samples:, :] = csum[:, window_samples:, :] - csum[:, :-window_samples, :]
        valid_count[:, window_samples:, :] = vcum[:, window_samples:, :] - vcum[:, :-window_samples, :]
        method_window = window_samples

    # Penalize hypotheses that frequently leave the map inside the evaluated
    # sequence window.
    window_len = np.minimum(np.arange(n_t) + 1, method_window)
    quality = valid_count / np.maximum(window_len[None, :, None], 1)
    summed = summed - (1.0 - quality) * 40.0

    pred = np.full(n_t, np.nan, dtype=float)
    chosen_scale = np.full(n_t, np.nan, dtype=float)
    chosen_start = np.full(n_t, np.nan, dtype=float)
    margin = np.full(n_t, np.nan, dtype=float)
    for k in range(n_t):
        flat = summed[:, k, :].reshape(-1)
        best_flat = int(np.argmax(flat))
        best_si, best_start = divmod(best_flat, n_s)
        scale = scales[best_si]
        delta_idx = int(round(direction_sign * scale * d_imu[k] / STEP_M))
        pos = best_start + delta_idx
        pos = int(np.clip(pos, 0, n_s - 1))
        pred[k] = dist[pos]
        chosen_scale[k] = scale
        chosen_start[k] = dist[best_start]
        top = np.partition(flat, -2)[-2:]
        margin[k] = float(top[-1] - top[-2]) if len(top) == 2 else math.nan
    meta = {
        "final_scale": float(chosen_scale[-1]),
        "median_scale": float(np.nanmedian(chosen_scale)),
        "final_start_m": float(chosen_start[-1]),
        "median_margin": float(np.nanmedian(margin)),
        "final_margin": float(margin[-1]),
        "window_samples": int(method_window),
    }
    return pred, meta


def eval_pred(pred: np.ndarray, truth: np.ndarray, stop: int | None = None) -> dict[str, float]:
    if stop is not None:
        pred = pred[:stop]
        truth = truth[:stop]
    warmup = min(20, max(0, len(pred) // 10))
    mask = np.isfinite(pred) & np.isfinite(truth)
    mask[:warmup] = False
    err = pred[mask] - truth[mask]
    return {
        "sample_count": int(len(err)),
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
        results.groupby(["metric_set", "method"])
        .agg(
            segment_count=("segment_label", "size"),
            median_abs_error_m=("median_abs_error_m", "median"),
            mean_abs_error_m=("mean_abs_error_m", "mean"),
            rmse_m=("rmse_m", "mean"),
            p90_abs_error_m=("p90_abs_error_m", "mean"),
            median_final_error_m=("final_abs_error_m", "median"),
            mean_final_error_m=("final_abs_error_m", "mean"),
            max_final_error_m=("final_abs_error_m", "max"),
            mean_within_25m_rate=("within_25m_rate", "mean"),
            mean_within_50m_rate=("within_50m_rate", "mean"),
        )
        .reset_index()
        .sort_values(["metric_set", "median_abs_error_m", "mean_abs_error_m"])
    )


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    trimmed = summary[summary["metric_set"] == "trim_reversal_tail"].sort_values("median_abs_error_m")
    fig, ax = plt.subplots(figsize=(11, 5), dpi=180)
    x = np.arange(len(trimmed))
    ax.bar(x - 0.2, trimmed["median_abs_error_m"], width=0.4, label="sample median")
    ax.bar(x + 0.2, trimmed["median_final_error_m"], width=0.4, label="endpoint median")
    ax.set_xticks(x)
    ax.set_xticklabels(trimmed["method"], rotation=30, ha="right")
    ax.set_ylabel("error / m")
    ax.set_title("Weak-mileage sequence filter, trimmed monotonic-pass evaluation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    refs, _ = arh.build_candidate_refs()
    ref = refs["forward_only"]
    queries = hmm.read_query_segments(AXIS_VARIANT, SAMPLE_PERIOD)
    scales = [0.45, 0.60, 0.75, 0.90, 1.05, 1.25, 1.50, 1.80, 2.20, 2.70, 3.20]
    windows = {
        "W120s": 30,
        "W240s": 60,
        "Wall": None,
    }
    rows = []
    meta_rows = []
    traj_rows = []
    for q in queries:
        stop, trim_meta = trim_stop_index(q)
        t_s = (pd.to_datetime(q.time).astype("int64").to_numpy(float) - pd.Timestamp(q.time[0]).value) / 1e9
        keep_idx = np.linspace(0, len(q.time) - 1, min(220, len(q.time))).round().astype(int)
        for label, window in windows.items():
            pred, meta = weak_mileage_filter(q, ref, scales, window)
            method = f"WeakMileage_{label}"
            meta_rows.append({"method": method, "segment_label": q.label, "direction": q.direction, **meta, **trim_meta})
            rows.append({"metric_set": "full_segment", "method": method, "segment_label": q.label, "direction": q.direction, **eval_pred(pred, q.truth_s)})
            rows.append({"metric_set": "trim_reversal_tail", "method": method, "segment_label": q.label, "direction": q.direction, **eval_pred(pred, q.truth_s, stop=stop)})
            if label in {"W240s", "Wall"}:
                for i in keep_idx:
                    traj_rows.append(
                        {
                            "method": method,
                            "segment_label": q.label,
                            "segment_short": q.label.replace("BMAW15230010L_", ""),
                            "direction": q.direction,
                            "time_s": float(t_s[i]),
                            "truth_s_m": float(q.truth_s[i]),
                            "pred_s_m": float(pred[i]),
                        }
                    )
    results = pd.DataFrame(rows)
    summary = summarize(results)
    meta = pd.DataFrame(meta_rows)
    traj = pd.DataFrame(traj_rows)
    results.to_csv(OUT_DIR / "weak_mileage_sequence_filter_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "weak_mileage_sequence_filter_summary.csv", index=False, encoding="utf-8-sig")
    meta.to_csv(OUT_DIR / "weak_mileage_sequence_filter_meta.csv", index=False, encoding="utf-8-sig")
    traj.to_csv(OUT_DIR / "weak_mileage_sequence_filter_trajectories.csv", index=False, encoding="utf-8-sig")
    plot_summary(summary, OUT_DIR / "weak_mileage_sequence_filter_summary.png")
    (OUT_DIR / "weak_mileage_sequence_filter_summary.json").write_text(
        json.dumps(
            {
                "scales": scales,
                "windows": windows,
                "summary": summary.to_dict(orient="records"),
                "meta": meta.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.round(3).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
