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


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\fixed_lag_online_hmm_experiment")
AXIS_VARIANT = "fwd_z_y_x_back_z_y_minusx"
SAMPLE_PERIOD = "4s"
STEP_M = 0.5


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def compute_viterbi_tables(
    q: hmm.QuerySegment,
    ref: dict[str, np.ndarray],
    features: list[str],
    weights: dict[str, float],
    sigma: float,
    vmax_mps: float,
    robust: bool = True,
    info_gate: bool = False,
    gate_min_scale: float = 0.12,
    gate_offset: float = 0.03,
    gate_span: float = 0.20,
    start_center_m: float | None = None,
    start_sigma_m: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    dist = ref["distance_m"]
    n_s = len(dist)
    ll = hmm.measurement_loglikelihood(q, ref, features, weights, sigma=sigma, robust=robust)
    margins = np.full(len(q.time), np.nan)
    if info_gate:
        ll, margins = hmm.apply_likelihood_uniqueness_gate(
            ll,
            dist,
            min_scale=gate_min_scale,
            offset=gate_offset,
            span=gate_span,
        )
    times = pd.to_datetime(q.time)
    ts = times.astype("int64").to_numpy(float) / 1e9
    direction_sign = 1 if q.direction == "forward" else -1
    dp = np.full((len(q.time), n_s), -np.inf, dtype=np.float32)
    prev = np.full((len(q.time), n_s), -1, dtype=np.int32)
    if start_center_m is None:
        dp[0] = ll[0]
    else:
        prior = -0.5 * ((dist - start_center_m) / max(start_sigma_m, 1e-3)) ** 2
        dp[0] = ll[0] + prior.astype(np.float32)
    for k in range(1, len(q.time)):
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
    meta = {
        "final_score_margin": float(np.nanmax(dp[-1]) - np.nanpercentile(dp[-1], 99)) if np.isfinite(dp[-1]).any() else math.nan,
        "median_measurement_margin": float(np.nanmedian(margins)),
    }
    return dp, prev, dist, meta


def backtrack_to(prev: np.ndarray, end_time: int, end_state: int, target_time: int) -> int:
    state = int(end_state)
    for k in range(end_time, target_time, -1):
        p = int(prev[k, state])
        if p < 0:
            break
        state = p
    return state


def fixed_lag_prediction(dp: np.ndarray, prev: np.ndarray, dist: np.ndarray, lag_samples: int) -> np.ndarray:
    n_t = dp.shape[0]
    pred_idx = np.zeros(n_t, dtype=int)
    for t in range(n_t):
        end_t = min(n_t - 1, t + lag_samples)
        if not np.isfinite(dp[end_t]).any():
            pred_idx[t] = 0
            continue
        end_state = int(np.argmax(dp[end_t]))
        pred_idx[t] = backtrack_to(prev, end_t, end_state, t)
    return dist[pred_idx]


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


def configs() -> list[dict]:
    return [
        {
            "family": "total",
            "label": "Total_vmax1.0",
            "features": ["total_raw_hp_z"],
            "weights": {"total_raw_hp_z": 1.0},
            "sigma": 1.2,
            "vmax_mps": 1.0,
            "info_gate": False,
        },
        {
            "family": "total",
            "label": "Total_vmax1.2",
            "features": ["total_raw_hp_z"],
            "weights": {"total_raw_hp_z": 1.0},
            "sigma": 1.2,
            "vmax_mps": 1.2,
            "info_gate": False,
        },
        {
            "family": "total",
            "label": "Total_vmax1.4",
            "features": ["total_raw_hp_z"],
            "weights": {"total_raw_hp_z": 1.0},
            "sigma": 1.2,
            "vmax_mps": 1.4,
            "info_gate": False,
        },
        {
            "family": "axis",
            "label": "AxisMidGate",
            "features": ["axis_x_hp_z", "axis_y_hp_z", "axis_total_hp_z"],
            "weights": {"axis_x_hp_z": 0.8, "axis_y_hp_z": 0.8, "axis_total_hp_z": 1.0},
            "sigma": 1.35,
            "vmax_mps": 1.4,
            "info_gate": True,
            "gate_min_scale": 0.30,
            "gate_offset": 0.02,
            "gate_span": 0.24,
        },
    ]


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
    full = summary[summary["metric_set"] == "trim_reversal_tail"].sort_values("median_abs_error_m")
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=180)
    x = np.arange(len(full))
    ax.bar(x - 0.2, full["median_abs_error_m"], width=0.4, label="sample median")
    ax.bar(x + 0.2, full["median_final_error_m"], width=0.4, label="endpoint median")
    ax.set_xticks(x)
    ax.set_xticklabels(full["method"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("error / m")
    ax.set_title("Fixed-lag HMM, trimmed monotonic-pass evaluation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    refs, _ = arh.build_candidate_refs()
    total_ref = refs["forward_only"]
    axis_ref = hmm.build_reference(AXIS_VARIANT, "all")
    queries = hmm.read_query_segments(AXIS_VARIANT, SAMPLE_PERIOD)

    lags = [0, 5, 15]  # 0 s, 20 s, 60 s at 4 s sampling.
    metric_rows = []
    traj_rows = []
    config_meta = []
    for q in queries:
        stop, trim_meta = trim_stop_index(q)
        t_s = (pd.to_datetime(q.time).astype("int64").to_numpy(float) - pd.Timestamp(q.time[0]).value) / 1e9
        keep_idx = np.linspace(0, len(q.time) - 1, min(200, len(q.time))).round().astype(int)
        for cfg in configs():
            ref = total_ref if cfg["family"] == "total" else axis_ref
            dp, prev, dist, meta = compute_viterbi_tables(
                q,
                ref,
                cfg["features"],
                cfg["weights"],
                sigma=cfg["sigma"],
                vmax_mps=cfg["vmax_mps"],
                robust=True,
                info_gate=cfg["info_gate"],
                gate_min_scale=cfg.get("gate_min_scale", 0.12),
                gate_offset=cfg.get("gate_offset", 0.03),
                gate_span=cfg.get("gate_span", 0.20),
            )
            config_meta.append({"segment_label": q.label, "base_label": cfg["label"], **meta, **trim_meta})
            for lag in lags:
                pred = fixed_lag_prediction(dp, prev, dist, lag)
                method = f"{cfg['label']}_lag{lag * 4}s"
                metric_rows.append({"metric_set": "full_segment", "method": method, "segment_label": q.label, "direction": q.direction, **eval_pred(pred, q.truth_s)})
                metric_rows.append({"metric_set": "trim_reversal_tail", "method": method, "segment_label": q.label, "direction": q.direction, **eval_pred(pred, q.truth_s, stop=stop)})
                if method in {"Total_vmax1.4_lag20s", "Total_vmax1.4_lag60s", "AxisMidGate_lag60s"}:
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

    results = pd.DataFrame(metric_rows)
    summary = summarize(results)
    traj = pd.DataFrame(traj_rows)
    meta_df = pd.DataFrame(config_meta)
    results.to_csv(OUT_DIR / "fixed_lag_hmm_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "fixed_lag_hmm_summary.csv", index=False, encoding="utf-8-sig")
    traj.to_csv(OUT_DIR / "fixed_lag_hmm_trajectories.csv", index=False, encoding="utf-8-sig")
    meta_df.to_csv(OUT_DIR / "fixed_lag_hmm_meta.csv", index=False, encoding="utf-8-sig")
    plot_summary(summary, OUT_DIR / "fixed_lag_hmm_summary.png")
    (OUT_DIR / "fixed_lag_hmm_summary.json").write_text(
        json.dumps(
            {
                "lags_samples": lags,
                "lags_seconds": [lag * 4 for lag in lags],
                "summary": summary.to_dict(orient="records"),
                "meta": meta_df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.round(3).head(18).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
