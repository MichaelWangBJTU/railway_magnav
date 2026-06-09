from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anchor_reference_hmm_experiment as arh
import axis_calibrated_hmm_experiment as hmm
from fixed_lag_online_hmm_experiment import compute_viterbi_tables, eval_pred, fixed_lag_prediction
from verify_turnaround_and_trim import trim_stop_index


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\coarse_start_online_hmm_experiment")
AXIS_VARIANT = "fwd_z_y_x_back_z_y_minusx"
SAMPLE_PERIOD = "4s"


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def configs() -> list[dict]:
    out = []
    for vmax in [1.0, 1.2, 1.4]:
        out.append(
            {
                "base_label": f"Total_vmax{vmax:g}",
                "family": "total",
                "features": ["total_raw_hp_z"],
                "weights": {"total_raw_hp_z": 1.0},
                "sigma": 1.2,
                "vmax_mps": vmax,
                "info_gate": False,
            }
        )
    return out


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
    trimmed = summary[summary["metric_set"] == "trim_reversal_tail"].sort_values("median_abs_error_m").head(12)
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=180)
    x = np.arange(len(trimmed))
    ax.bar(x - 0.2, trimmed["median_abs_error_m"], width=0.4, label="sample median")
    ax.bar(x + 0.2, trimmed["median_final_error_m"], width=0.4, label="endpoint median")
    ax.set_xticks(x)
    ax.set_xticklabels(trimmed["method"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("error / m")
    ax.set_title("Coarse-start fixed-lag HMM, trimmed monotonic-pass evaluation")
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
    lags = [0, 5, 15]  # 0, 20, 60 s
    sigmas = [30.0, 60.0, 100.0]
    rows = []
    meta_rows = []
    traj_rows = []
    for q in queries:
        stop, trim_meta = trim_stop_index(q)
        start_center = float(q.truth_s[0])
        t_s = (pd.to_datetime(q.time).astype("int64").to_numpy(float) - pd.Timestamp(q.time[0]).value) / 1e9
        keep_idx = np.linspace(0, len(q.time) - 1, min(220, len(q.time))).round().astype(int)
        for sigma0 in sigmas:
            for cfg in configs():
                dp, prev, dist, meta = compute_viterbi_tables(
                    q,
                    ref,
                    cfg["features"],
                    cfg["weights"],
                    sigma=cfg["sigma"],
                    vmax_mps=cfg["vmax_mps"],
                    robust=True,
                    info_gate=cfg["info_gate"],
                    start_center_m=start_center,
                    start_sigma_m=sigma0,
                )
                meta_rows.append({"segment_label": q.label, "base_label": cfg["base_label"], "start_sigma_m": sigma0, **meta, **trim_meta})
                for lag in lags:
                    pred = fixed_lag_prediction(dp, prev, dist, lag)
                    method = f"{cfg['base_label']}_start{int(sigma0)}m_lag{lag * 4}s"
                    rows.append({"metric_set": "full_segment", "method": method, "segment_label": q.label, "direction": q.direction, **eval_pred(pred, q.truth_s)})
                    rows.append({"metric_set": "trim_reversal_tail", "method": method, "segment_label": q.label, "direction": q.direction, **eval_pred(pred, q.truth_s, stop=stop)})
                    if method in {"Total_vmax1.4_start60m_lag20s", "Total_vmax1.4_start60m_lag60s"}:
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
    results.to_csv(OUT_DIR / "coarse_start_online_hmm_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "coarse_start_online_hmm_summary.csv", index=False, encoding="utf-8-sig")
    meta.to_csv(OUT_DIR / "coarse_start_online_hmm_meta.csv", index=False, encoding="utf-8-sig")
    traj.to_csv(OUT_DIR / "coarse_start_online_hmm_trajectories.csv", index=False, encoding="utf-8-sig")
    plot_summary(summary, OUT_DIR / "coarse_start_online_hmm_summary.png")
    (OUT_DIR / "coarse_start_online_hmm_summary.json").write_text(
        json.dumps(
            {
                "start_prior": "centered at first GPGGA along-track coordinate; simulates known start / coarse GNSS initialization, not a no-initial-position method",
                "lags_seconds": [lag * 4 for lag in lags],
                "start_sigmas_m": sigmas,
                "summary": summary.to_dict(orient="records"),
                "meta": meta.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.round(3).head(24).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
