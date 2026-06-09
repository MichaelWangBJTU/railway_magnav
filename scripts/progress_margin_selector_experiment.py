from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\progress_margin_selector_experiment")


TOTAL_METHODS = [
    "TotalHP_vmax1_uniform",
    "TotalHP_vmax1.2_uniform",
    "TotalHP_vmax1.4_uniform",
]
AXIS_METHOD = "AxisCal_XY_TotalHP_MidGate_Viterbi"


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def progress_from_traj(traj: pd.DataFrame, segment: str, method: str) -> float:
    g = traj[(traj["segment_label"] == segment) & (traj["method"] == method)].sort_values("time")
    if g.empty:
        return math.nan
    return float(abs(g["pred_s_m"].iloc[-1] - g["pred_s_m"].iloc[0]))


def progress_compat(progress_m: float, imu_m: float) -> float:
    eps = 10.0
    if not np.isfinite(progress_m) or not np.isfinite(imu_m):
        return math.inf
    return float(abs(np.log((progress_m + eps) / (imu_m + eps))))


def choose_total(cands: list[dict]) -> dict:
    best_compat = min(c["progress_compat"] for c in cands)
    near = [c for c in cands if c["progress_compat"] <= best_compat + 0.04]
    # When progress evidence is indistinguishable, prefer the candidate with a
    # stronger final score margin; if margins are close, prefer the less
    # restrictive speed bound to avoid endpoint lag.
    max_margin = max(c["final_score_margin"] for c in near)
    near_margin = [c for c in near if c["final_score_margin"] >= max_margin - 0.20]
    return sorted(near_margin, key=lambda c: c["vmax_mps"], reverse=True)[0]


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("method")
        .agg(
            segment_count=("segment_label", "size"),
            median_abs_error_m=("median_abs_error_m", "median"),
            mean_abs_error_m=("mean_abs_error_m", "mean"),
            rmse_m=("rmse_m", "mean"),
            p90_abs_error_m=("p90_abs_error_m", "mean"),
            median_final_error_m=("final_abs_error_m", "median"),
            mean_final_error_m=("final_abs_error_m", "mean"),
            max_final_error_m=("final_abs_error_m", "max"),
        )
        .reset_index()
        .sort_values(["median_abs_error_m", "mean_abs_error_m", "mean_final_error_m"])
    )


def method_row(df: pd.DataFrame, segment: str, method: str) -> pd.Series:
    rows = df[(df["segment_label"] == segment) & (df["method"] == method)]
    if rows.empty:
        raise KeyError((segment, method))
    return rows.iloc[0]


def add_metric_row(rows: list[dict], label: str, source_row: pd.Series, chosen: str | None = None, extra: dict | None = None) -> None:
    row = {
        "method": label,
        "segment_label": source_row["segment_label"],
        "direction": source_row["direction"],
        "chosen_candidate": chosen or source_row["method"],
        "median_abs_error_m": float(source_row["median_abs_error_m"]),
        "mean_abs_error_m": float(source_row["mean_abs_error_m"]),
        "rmse_m": float(source_row["rmse_m"]),
        "p90_abs_error_m": float(source_row["p90_abs_error_m"]),
        "final_abs_error_m": float(source_row["final_abs_error_m"]),
    }
    if extra:
        row.update(extra)
    rows.append(row)


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=180)
    x = np.arange(len(summary))
    ax.bar(x - 0.18, summary["median_abs_error_m"], width=0.36, label="median sample")
    ax.bar(x + 0.18, summary["median_final_error_m"], width=0.36, label="median endpoint")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"], rotation=25, ha="right")
    ax.set_ylabel("error / m")
    ax.set_title("Progress-margin selector: sample and endpoint errors")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_selected_trajectories(decisions: pd.DataFrame, total_traj: pd.DataFrame, axis_traj: pd.DataFrame, path: Path) -> None:
    segments = decisions["segment_label"].tolist()
    fig, axes = plt.subplots(len(segments), 1, figsize=(12.5, 2.7 * len(segments)), dpi=180, sharex=False)
    if len(segments) == 1:
        axes = [axes]
    for ax, (_, decision) in zip(axes, decisions.iterrows()):
        segment = decision["segment_label"]
        chosen = decision["selected_candidate"]
        if chosen == "AxisMidGate":
            g = axis_traj[(axis_traj["segment_label"] == segment) & (axis_traj["method"] == AXIS_METHOD)].copy()
        else:
            g = total_traj[(total_traj["segment_label"] == segment) & (total_traj["method"] == chosen)].copy()
        g = g.sort_values("time")
        t = (pd.to_datetime(g["time"]).astype("int64").to_numpy(float) - pd.to_datetime(g["time"].iloc[0]).value) / 1e9
        ax.plot(t, g["truth_s_m"], color="black", lw=1.8, label="SPAN/GPGGA truth")
        ax.plot(t, g["pred_s_m"], color="#d95f02", lw=1.3, label=f"selected: {chosen}")
        ax.set_title(f"{segment.replace('BMAW15230010L_', '')} / {decision['direction']}")
        ax.set_ylabel("s / m")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=7)
    axes[-1].set_xlabel("segment time / s")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def selected_trajectory_rows(decisions: pd.DataFrame, total_traj: pd.DataFrame, axis_traj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, decision in decisions.iterrows():
        segment = decision["segment_label"]
        chosen = decision["selected_candidate"]
        if chosen == "AxisMidGate":
            g = axis_traj[(axis_traj["segment_label"] == segment) & (axis_traj["method"] == AXIS_METHOD)].copy()
            plotted_method = AXIS_METHOD
        else:
            g = total_traj[(total_traj["segment_label"] == segment) & (total_traj["method"] == chosen)].copy()
            plotted_method = chosen
        g = g.sort_values("time").copy()
        if g.empty:
            continue
        times = pd.to_datetime(g["time"])
        g["segment_time_s"] = (times.astype("int64").to_numpy(float) - times.iloc[0].value) / 1e9
        g["selected_candidate"] = chosen
        g["plotted_method"] = plotted_method
        g["selected_source"] = decision["selected_source"]
        g["figure_label"] = segment.replace("BMAW15230010L_", "")
        rows.append(
            g[
                [
                    "figure_label",
                    "segment_label",
                    "direction",
                    "selected_candidate",
                    "selected_source",
                    "plotted_method",
                    "time",
                    "segment_time_s",
                    "truth_s_m",
                    "pred_s_m",
                    "error_m",
                ]
            ]
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "figure_label",
                "segment_label",
                "direction",
                "selected_candidate",
                "selected_source",
                "plotted_method",
                "time",
                "segment_time_s",
                "truth_s_m",
                "pred_s_m",
                "error_m",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def run() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_res = pd.read_csv("forward_anchor_hmm_tuning/forward_anchor_hmm_tuning_results.csv")
    total_traj = pd.read_csv("forward_anchor_hmm_tuning/forward_anchor_hmm_tuning_trajectories.csv")
    axis_res = pd.read_csv("axis_calibrated_hmm_gate_sweep/axis_calibrated_hmm_results.csv")
    axis_traj = pd.read_csv("axis_calibrated_hmm_gate_sweep/axis_calibrated_hmm_trajectories.csv")
    imu_res = pd.read_csv("imu_progress_gated_ensemble/imu_progress_gated_ensemble_results.csv")
    endpoint_res = pd.read_csv("imu_progress_gated_ensemble/endpoint_evaluation/endpoint_error_by_segment.csv")

    axis_rows = axis_res[axis_res["method"] == AXIS_METHOD].copy()
    axis_by_seg = axis_rows.set_index("segment_label")
    imu_by_seg = imu_res[imu_res["method"] == "IMUProgressClosest_TotalVsAxis"].set_index("segment_label")

    metric_rows = []
    decision_rows = []
    segments = list(dict.fromkeys(total_res["segment_label"].tolist()))
    for segment in segments:
        imu_m = float(imu_by_seg.loc[segment, "imu_distance_m"])
        total_candidates = []
        for method in TOTAL_METHODS:
            r = method_row(total_res, segment, method)
            progress = progress_from_traj(total_traj, segment, method)
            total_candidates.append(
                {
                    "method": method,
                    "vmax_mps": float(r["vmax_mps"]),
                    "progress_m": progress,
                    "progress_compat": progress_compat(progress, imu_m),
                    "final_score_margin": float(r["final_score_margin"]),
                    "row": r,
                }
            )
        total_choice = choose_total(total_candidates)

        axis_r = axis_by_seg.loc[segment]
        axis_progress = float(imu_res[(imu_res["segment_label"] == segment) & (imu_res["method"] == "AxisAllMidGate")]["axis_progress_m"].iloc[0])
        axis_compat = progress_compat(axis_progress, imu_m)
        axis_margin = float(axis_r["final_score_margin"])
        choose_axis = axis_margin >= 5.0 and (axis_compat + 0.04 < total_choice["progress_compat"])
        if choose_axis:
            chosen_label = "AxisMidGate"
            chosen_row = axis_r.copy()
            chosen_row["segment_label"] = segment
            chosen_source = "axis"
        else:
            chosen_label = total_choice["method"]
            chosen_row = total_choice["row"]
            chosen_source = "total"

        add_metric_row(metric_rows, "ProgressMarginSelector", chosen_row, chosen_label, {"chosen_source": chosen_source})
        add_metric_row(metric_rows, "FixedTotal_vmax1.2", method_row(total_res, segment, "TotalHP_vmax1.2_uniform"))
        # Reuse the already validated IMU-progress ensemble metrics for comparison.
        ens = method_row(endpoint_res, segment, "IMUProgressClosest_TotalVsAxis")
        metric_rows.append(
            {
                "method": "IMUProgressClosest_TotalVsAxis",
                "segment_label": ens["segment_label"],
                "direction": ens["direction"],
                "chosen_candidate": ens["selected_candidate"],
                "median_abs_error_m": float(ens["median_abs_error_m"]),
                "mean_abs_error_m": float(ens["mean_abs_error_m"]),
                "rmse_m": float(ens["rmse_m"]),
                "p90_abs_error_m": float(ens["p90_abs_error_m"]),
                "final_abs_error_m": float(ens["final_error_m"]),
                "chosen_source": "previous_ensemble",
            }
        )
        decision_rows.append(
            {
                "segment_label": segment,
                "direction": chosen_row["direction"],
                "imu_distance_m": imu_m,
                "selected_candidate": chosen_label,
                "selected_source": chosen_source,
                "selected_progress_compat": axis_compat if choose_axis else total_choice["progress_compat"],
                "selected_final_score_margin": axis_margin if choose_axis else total_choice["final_score_margin"],
                "axis_progress_compat": axis_compat,
                "axis_final_score_margin": axis_margin,
                "best_total_candidate": total_choice["method"],
                "best_total_progress_compat": total_choice["progress_compat"],
                "best_total_final_score_margin": total_choice["final_score_margin"],
            }
        )

    results = pd.DataFrame(metric_rows)
    decisions = pd.DataFrame(decision_rows)
    summary = summarize(results)
    results.to_csv(OUT_DIR / "progress_margin_selector_results.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(OUT_DIR / "progress_margin_selector_decisions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "progress_margin_selector_summary.csv", index=False, encoding="utf-8-sig")
    selected_trajectory_rows(decisions, total_traj, axis_traj).to_csv(
        OUT_DIR / "progress_margin_selector_selected_trajectories.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_summary(summary, OUT_DIR / "progress_margin_selector_summary.png")
    plot_selected_trajectories(decisions, total_traj, axis_traj, OUT_DIR / "progress_margin_selector_selected_trajectories.png")
    (OUT_DIR / "progress_margin_selector_summary.json").write_text(
        json.dumps(
            {
                "rule": {
                    "total_choice": "among vmax 1.0/1.2/1.4, keep candidates within 0.04 log-progress mismatch of best; prefer final_score_margin within 0.20 of max, then higher vmax",
                    "axis_choice": "choose axis only when axis final_score_margin >= 5 and axis log-progress mismatch is at least 0.04 better than selected total",
                },
                "summary": summary.to_dict(orient="records"),
                "decisions": decisions.to_dict(orient="records"),
                "results": results.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Decisions:")
    print(decisions.round(3).to_string(index=False))
    print("\nSummary:")
    print(summary.round(3).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
