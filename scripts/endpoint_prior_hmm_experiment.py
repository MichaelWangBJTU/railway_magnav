from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anchor_reference_hmm_experiment as arh
import axis_calibrated_hmm_experiment as hmm


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\endpoint_prior_hmm_experiment")
AXIS_VARIANT = "fwd_z_y_x_back_z_y_minusx"
SAMPLE_PERIOD = "4s"
FEATURES = ["total_raw_hp_z"]
WEIGHTS = {"total_raw_hp_z": 1.0}


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def eval_path(pred: np.ndarray, truth: np.ndarray, warmup: int = 10) -> dict[str, float]:
    mask = np.isfinite(pred) & np.isfinite(truth)
    if warmup:
        mask[: min(warmup, len(mask))] = False
    err = pred[mask] - truth[mask]
    if err.size == 0:
        return {
            "sample_count": 0,
            "median_abs_error_m": math.nan,
            "mean_abs_error_m": math.nan,
            "rmse_m": math.nan,
            "p90_abs_error_m": math.nan,
            "final_abs_error_m": math.nan,
        }
    return {
        "sample_count": int(err.size),
        "median_abs_error_m": float(np.median(np.abs(err))),
        "mean_abs_error_m": float(np.mean(np.abs(err))),
        "rmse_m": float(np.sqrt(np.mean(err * err))),
        "p90_abs_error_m": float(np.percentile(np.abs(err), 90)),
        "final_abs_error_m": float(abs(err[-1])),
    }


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in results.groupby("method"):
        rows.append(
            {
                "method": method,
                "segment_count": int(len(g)),
                "median_abs_error_m": float(g["median_abs_error_m"].median()),
                "mean_abs_error_m": float(g["mean_abs_error_m"].mean()),
                "rmse_m": float(g["rmse_m"].mean()),
                "p90_abs_error_m": float(g["p90_abs_error_m"].mean()),
                "median_final_abs_error_m": float(g["final_abs_error_m"].median()),
                "mean_final_abs_error_m": float(g["final_abs_error_m"].mean()),
                "max_final_abs_error_m": float(g["final_abs_error_m"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["median_abs_error_m", "mean_abs_error_m"])


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    top = summary.head(16).copy()
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=180)
    x = np.arange(len(top))
    ax.bar(x - 0.2, top["median_abs_error_m"], width=0.4, label="median")
    ax.bar(x + 0.2, top["mean_final_abs_error_m"], width=0.4, label="endpoint mean")
    ax.set_xticks(x)
    ax.set_xticklabels(top["method"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("error / m")
    ax.set_title("Endpoint-prior HMM: practical start prior without GNSS along the route")
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

    rows = []
    traj_rows = []
    configs = [
        ("uniform", 1.2, False, 0.0),
        ("uniform", 1.4, False, 0.0),
        ("endpoint_by_direction", 1.0, False, 0.0),
        ("endpoint_by_direction", 1.2, False, 0.0),
        ("endpoint_by_direction", 1.4, False, 0.0),
        ("endpoint_by_direction", 1.6, False, 0.0),
        ("endpoint_by_direction", 1.2, True, 0.04),
        ("endpoint_by_direction", 1.4, True, 0.04),
    ]

    for q in queries:
        for start_prior, vmax, speed_prior, speed_weight in configs:
            method = (
                f"start-{start_prior}_vmax{vmax:g}"
                + ("_speed" + str(speed_weight).replace(".", "p") if speed_prior else "")
            )
            pred, meta = hmm.viterbi_track(
                q,
                ref,
                FEATURES,
                WEIGHTS,
                sigma=1.2,
                vmax_mps=vmax,
                robust=True,
                info_gate=False,
                speed_prior=speed_prior,
                speed_weight=speed_weight,
                speed_sigma_mps=0.35,
                start_prior=start_prior,
            )
            rows.append(
                {
                    "method": method,
                    "segment_label": q.label,
                    "direction": q.direction,
                    "start_prior": start_prior,
                    "vmax_mps": vmax,
                    "speed_prior": int(speed_prior),
                    "speed_weight": speed_weight,
                    **eval_path(pred, q.truth_s),
                    **meta,
                }
            )
            take = np.linspace(0, len(pred) - 1, min(220, len(pred))).round().astype(int)
            for i in take:
                traj_rows.append(
                    {
                        "method": method,
                        "segment_label": q.label,
                        "time_index": int(i),
                        "truth_s_m": float(q.truth_s[i]),
                        "pred_s_m": float(pred[i]),
                        "error_m": float(pred[i] - q.truth_s[i]),
                    }
                )

    results = pd.DataFrame(rows)
    summary = summarize(results)
    trajectories = pd.DataFrame(traj_rows)

    results.to_csv(OUT_DIR / "endpoint_prior_hmm_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "endpoint_prior_hmm_summary.csv", index=False, encoding="utf-8-sig")
    trajectories.to_csv(OUT_DIR / "endpoint_prior_hmm_trajectories.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "endpoint_prior_hmm_summary.json").write_text(
        json.dumps(
            {"summary": summary.to_dict(orient="records"), "results": results.to_dict(orient="records")},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_summary(summary, OUT_DIR / "endpoint_prior_hmm_summary.png")

    lines = [
        "# Endpoint-prior HMM Experiment",
        "",
        "Purpose: test a practical rail-operation prior. The vehicle is assumed to start near one endpoint of the mapped rail section, but no along-route GNSS truth is used by the matcher.",
        "",
        "This is not the same as using SPAN/GPGGA for navigation. It is closer to a station/depot departure prior and is therefore defensible for railway scenarios.",
        "",
        "Aggregate summary:",
        "",
        summary.head(20).to_markdown(index=False, floatfmt=".3f"),
        "",
        "Interpretation placeholder:",
        "",
        "- If endpoint prior greatly improves uniform-start HMM, the publishable online story should include a practical start prior.",
        "- If it does not improve endpoint robustness, the main blocker is not cold-start only; it is transition/progress modeling and magnetic ambiguity.",
    ]
    (OUT_DIR / "endpoint_prior_hmm_notes.md").write_text("\n".join(lines), encoding="utf-8")

    print(summary.head(20).round(3).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
