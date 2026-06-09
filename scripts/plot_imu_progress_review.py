from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anchor_reference_hmm_experiment as arh
import axis_calibrated_hmm_experiment as hmm
import imu_progress_gated_ensemble as ipe


ROOT = Path(r"C:\Users\m1352\Documents\railway_magnav")
OUT_DIR = ROOT / "imu_progress_gated_ensemble" / "review_figures"
AXIS_VARIANT = "fwd_z_y_x_back_z_y_minusx"
SAMPLE_PERIOD = "4s"


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def short_label(seg: str) -> str:
    return seg.replace("BMAW15230010L_", "")


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(ROOT / "imu_progress_gated_ensemble" / "imu_progress_gated_ensemble_results.csv")
    summary = pd.read_csv(ROOT / "imu_progress_gated_ensemble" / "imu_progress_gated_ensemble_summary.csv")
    return results, summary


def plot_metric_summary(summary: pd.DataFrame) -> Path:
    all_set = summary[summary["summary_set"] == "all_segments"].copy()
    order = ["TotalForwardAnchor", "AxisAllMidGate", "IMUProgressClosest_TotalVsAxis"]
    all_set["method"] = pd.Categorical(all_set["method"], categories=order, ordered=True)
    all_set = all_set.sort_values("method")
    metrics = [
        ("median_abs_error_m", "中位误差"),
        ("mean_abs_error_m", "平均误差"),
        ("rmse_m", "RMSE"),
    ]
    x = np.arange(len(metrics))
    width = 0.24
    colors = {
        "TotalForwardAnchor": "#2a6fbb",
        "AxisAllMidGate": "#e07a2f",
        "IMUProgressClosest_TotalVsAxis": "#2b9348",
    }
    names = {
        "TotalForwardAnchor": "总场锚点",
        "AxisAllMidGate": "轴校准",
        "IMUProgressClosest_TotalVsAxis": "IMU进度门控",
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=180)
    for i, (_, row) in enumerate(all_set.iterrows()):
        values = [row[m[0]] for m in metrics]
        ax.bar(x + (i - 1) * width, values, width, label=names[str(row["method"])], color=colors[str(row["method"])])
        for xx, yy in zip(x + (i - 1) * width, values):
            ax.text(xx, yy + 2.0, f"{yy:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("误差 / m")
    ax.set_title("全段口径：三种方法总体指标对比")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = OUT_DIR / "01_method_metric_summary.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_segment_errors(results: pd.DataFrame) -> Path:
    data = results[results["method"].isin(["TotalForwardAnchor", "AxisAllMidGate", "IMUProgressClosest_TotalVsAxis"])].copy()
    order = data["segment_label"].drop_duplicates().tolist()
    methods = ["TotalForwardAnchor", "AxisAllMidGate", "IMUProgressClosest_TotalVsAxis"]
    names = ["总场锚点", "轴校准", "IMU门控"]
    colors = ["#2a6fbb", "#e07a2f", "#2b9348"]
    x = np.arange(len(order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=180)
    for i, method in enumerate(methods):
        vals = []
        for seg in order:
            vals.append(float(data[(data["method"] == method) & (data["segment_label"] == seg)]["median_abs_error_m"].iloc[0]))
        ax.bar(x + (i - 1) * width, vals, width, label=names[i], color=colors[i])
        for xx, yy in zip(x + (i - 1) * width, vals):
            ax.text(xx, yy + 4.0, f"{yy:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.axvspan(1 - 0.5, 1 + 0.5, color="#f4cccc", alpha=0.25, label="严重真值轴异常")
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(s) for s in order])
    ax.set_ylabel("中位绝对误差 / m")
    ax.set_title("分段误差对比：IMU门控在难例中选择更合适候选")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    path = OUT_DIR / "02_segment_median_error_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_progress_gate(results: pd.DataFrame) -> Path:
    data = results[results["method"] == "IMUProgressClosest_TotalVsAxis"].copy()
    x = np.arange(len(data))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 5.4), dpi=180)
    ax.bar(x - width, data["imu_distance_m"], width, label="IMU积分进度", color="#555555")
    ax.bar(x, data["total_progress_m"], width, label="总场候选进度", color="#2a6fbb")
    ax.bar(x + width, data["axis_progress_m"], width, label="轴候选进度", color="#e07a2f")
    for i, row in enumerate(data.itertuples()):
        chosen_x = x[i] if row.chosen_candidate == "TotalForwardAnchor" else x[i] + width
        chosen_y = row.total_progress_m if row.chosen_candidate == "TotalForwardAnchor" else row.axis_progress_m
        ax.scatter([chosen_x], [chosen_y + 18], marker="v", s=55, color="#2b9348", zorder=5)
        ax.text(chosen_x, chosen_y + 35, "选中", ha="center", va="bottom", fontsize=8, color="#2b9348")
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(s) for s in data["segment_label"]])
    ax.set_ylabel("整段进度 / m")
    ax.set_title("IMU弱进度门控：选择更接近速度积分量级的候选轨迹")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    path = OUT_DIR / "03_imu_progress_gate_diagnostic.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def recompute_trajectories() -> pd.DataFrame:
    refs, _ = arh.build_candidate_refs()
    ref_total = refs["forward_only"]
    ref_axis = hmm.build_reference(AXIS_VARIANT, "all")
    queries = hmm.read_query_segments(AXIS_VARIANT, SAMPLE_PERIOD)
    rows = []
    for q in queries:
        pred_total = ipe.total_candidate(q, ref_total)
        pred_axis = ipe.axis_candidate(q, ref_axis)
        imu_dist = ipe.imu_distance(q)
        total_compat = ipe.progress_compatibility(abs(pred_total[-1] - pred_total[0]), imu_dist)
        axis_compat = ipe.progress_compatibility(abs(pred_axis[-1] - pred_axis[0]), imu_dist)
        selected_name = "AxisAllMidGate" if axis_compat < total_compat else "TotalForwardAnchor"
        pred_selected = pred_axis if selected_name == "AxisAllMidGate" else pred_total
        t0 = pd.Timestamp(q.time[0])
        for i in range(len(q.time)):
            rows.append(
                {
                    "segment_label": q.label,
                    "direction": q.direction,
                    "time_s": (pd.Timestamp(q.time[i]) - t0).total_seconds(),
                    "truth_s_m": float(q.truth_s[i]),
                    "total_pred_m": float(pred_total[i]),
                    "axis_pred_m": float(pred_axis[i]),
                    "selected_pred_m": float(pred_selected[i]),
                    "selected_candidate": selected_name,
                }
            )
    traj = pd.DataFrame(rows)
    traj.to_csv(OUT_DIR / "recomputed_candidate_trajectories.csv", index=False, encoding="utf-8-sig")
    return traj


def plot_trajectories(traj: pd.DataFrame) -> Path:
    segs = traj["segment_label"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(segs), 1, figsize=(13, 16), dpi=180, sharex=False)
    if len(segs) == 1:
        axes = [axes]
    for ax, seg in zip(axes, segs):
        part = traj[traj["segment_label"] == seg]
        selected_name = str(part["selected_candidate"].iloc[0])
        ax.plot(part["time_s"], part["truth_s_m"], color="black", lw=1.8, label="SPAN/GPGGA评价真值")
        ax.plot(part["time_s"], part["total_pred_m"], color="#2a6fbb", lw=1.0, alpha=0.85, label="总场锚点候选")
        ax.plot(part["time_s"], part["axis_pred_m"], color="#e07a2f", lw=1.0, alpha=0.85, label="轴校准候选")
        ax.plot(part["time_s"], part["selected_pred_m"], color="#2b9348", lw=1.5, linestyle="--", label=f"门控选中: {selected_name}")
        ax.set_title(f"{short_label(seg)} / {part['direction'].iloc[0]}", loc="left", fontsize=10)
        ax.set_ylabel("沿轨距离 / m")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("本段相对时间 / s")
    axes[0].legend(loc="upper right", ncol=2, frameon=True, fontsize=8)
    fig.suptitle("逐段轨迹对比：真值、两个磁匹配候选与 IMU 门控选中轨迹", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.5)
    path = OUT_DIR / "04_selected_trajectory_by_segment.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_error_timeseries(traj: pd.DataFrame) -> Path:
    segs = traj["segment_label"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(segs), 1, figsize=(13, 16), dpi=180, sharex=False)
    if len(segs) == 1:
        axes = [axes]
    for ax, seg in zip(axes, segs):
        part = traj[traj["segment_label"] == seg].copy()
        ax.plot(part["time_s"], part["total_pred_m"] - part["truth_s_m"], color="#2a6fbb", lw=1.0, alpha=0.85, label="总场误差")
        ax.plot(part["time_s"], part["axis_pred_m"] - part["truth_s_m"], color="#e07a2f", lw=1.0, alpha=0.85, label="轴候选误差")
        ax.plot(part["time_s"], part["selected_pred_m"] - part["truth_s_m"], color="#2b9348", lw=1.5, linestyle="--", label="门控误差")
        ax.axhline(0, color="black", lw=0.8)
        ax.axhspan(-25, 25, color="#d9ead3", alpha=0.25)
        ax.set_title(f"{short_label(seg)} / {part['direction'].iloc[0]}", loc="left", fontsize=10)
        ax.set_ylabel("误差 / m")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("本段相对时间 / s")
    axes[0].legend(loc="lower left", ncol=3, frameon=True, fontsize=8)
    fig.suptitle("逐段误差曲线：绿色区域为 ±25 m", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.5)
    path = OUT_DIR / "05_error_timeseries_by_segment.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results, summary = load_results()
    paths = [
        plot_metric_summary(summary),
        plot_segment_errors(results),
        plot_progress_gate(results),
    ]
    traj = recompute_trajectories()
    paths += [plot_trajectories(traj), plot_error_timeseries(traj)]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
