from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\m1352\Documents\railway_magnav")
TRAJ = ROOT / "imu_progress_gated_ensemble" / "review_figures" / "recomputed_candidate_trajectories.csv"
OUT_DIR = ROOT / "imu_progress_gated_ensemble" / "endpoint_evaluation"


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def short_label(seg: str) -> str:
    return seg.replace("BMAW15230010L_", "")


def evaluate() -> pd.DataFrame:
    df = pd.read_csv(TRAJ)
    methods = [
        ("TotalForwardAnchor", "total_pred_m"),
        ("AxisAllMidGate", "axis_pred_m"),
        ("IMUProgressClosest_TotalVsAxis", "selected_pred_m"),
    ]
    rows = []
    for (seg, direction), g in df.groupby(["segment_label", "direction"], sort=False):
        truth = g["truth_s_m"].to_numpy(float)
        for method, col in methods:
            pred = g[col].to_numpy(float)
            err = pred - truth
            rows.append(
                {
                    "segment_label": seg,
                    "segment_short": short_label(seg),
                    "direction": direction,
                    "method": method,
                    "start_error_m": float(abs(err[0])),
                    "final_error_m": float(abs(err[-1])),
                    "net_progress_error_m": float(abs(abs(pred[-1] - pred[0]) - abs(truth[-1] - truth[0]))),
                    "median_abs_error_m": float(np.median(np.abs(err))),
                    "mean_abs_error_m": float(np.mean(np.abs(err))),
                    "rmse_m": float(np.sqrt(np.mean(err * err))),
                    "max_abs_error_m": float(np.max(np.abs(err))),
                    "p90_abs_error_m": float(np.percentile(np.abs(err), 90)),
                    "within_25m_rate": float(np.mean(np.abs(err) <= 25.0)),
                    "within_50m_rate": float(np.mean(np.abs(err) <= 50.0)),
                    "selected_candidate": str(g["selected_candidate"].iloc[0]) if method == "IMUProgressClosest_TotalVsAxis" else method,
                }
            )
    return pd.DataFrame(rows)


def summarize(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in eval_df.groupby("method"):
        rows.append(
            {
                "method": method,
                "segment_count": int(len(g)),
                "median_final_error_m": float(g["final_error_m"].median()),
                "mean_final_error_m": float(g["final_error_m"].mean()),
                "max_final_error_m": float(g["final_error_m"].max()),
                "median_max_error_m": float(g["max_abs_error_m"].median()),
                "mean_within_25m_rate": float(g["within_25m_rate"].mean()),
                "mean_within_50m_rate": float(g["within_50m_rate"].mean()),
                "median_net_progress_error_m": float(g["net_progress_error_m"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("median_final_error_m")


def plot_final_errors(eval_df: pd.DataFrame) -> Path:
    order = eval_df["segment_label"].drop_duplicates().tolist()
    methods = ["TotalForwardAnchor", "AxisAllMidGate", "IMUProgressClosest_TotalVsAxis"]
    labels = ["总场锚点", "轴校准", "IMU门控"]
    colors = ["#2a6fbb", "#e07a2f", "#2b9348"]
    x = np.arange(len(order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5.3), dpi=180)
    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        vals = [
            float(eval_df[(eval_df["segment_label"] == seg) & (eval_df["method"] == method)]["final_error_m"].iloc[0])
            for seg in order
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)
        for xx, yy in zip(x + (i - 1) * width, vals):
            ax.text(xx, yy + 3, f"{yy:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.axvspan(1 - 0.5, 1 + 0.5, color="#f4cccc", alpha=0.25, label="严重真值轴异常")
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(s) for s in order])
    ax.set_ylabel("终点绝对误差 / m")
    ax.set_title("完整段从头导航到尾端：最终位置误差")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    path = OUT_DIR / "endpoint_final_error_by_segment.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_coverage(eval_df: pd.DataFrame) -> Path:
    data = eval_df[eval_df["method"] == "IMUProgressClosest_TotalVsAxis"].copy()
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=180)
    ax.bar(x - 0.18, data["within_25m_rate"] * 100, 0.36, label="|误差|≤25 m", color="#2b9348")
    ax.bar(x + 0.18, data["within_50m_rate"] * 100, 0.36, label="|误差|≤50 m", color="#6ab04c")
    ax.set_xticks(x)
    ax.set_xticklabels(data["segment_short"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("时间覆盖率 / %")
    ax.set_title("IMU门控方法：完整段误差覆盖率")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = OUT_DIR / "selected_method_error_coverage.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_notes(eval_df: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    selected = eval_df[eval_df["method"] == "IMUProgressClosest_TotalVsAxis"].copy()
    lines = [
        "# Endpoint Error Evaluation",
        "",
        "Purpose: evaluate complete-segment navigation from the beginning to the end, not only median sample error.",
        "",
        "Aggregate endpoint summary:",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "IMU-progress-gated method by segment:",
        "",
        selected[
            [
                "segment_label",
                "direction",
                "selected_candidate",
                "start_error_m",
                "final_error_m",
                "net_progress_error_m",
                "median_abs_error_m",
                "max_abs_error_m",
                "within_25m_rate",
                "within_50m_rate",
            ]
        ].to_markdown(index=False, floatfmt=".3f"),
        "",
        "Interpretation:",
        "",
        "- Final endpoint error is a stricter usability metric than median error.",
        "- `1_seg03` should be reported separately because the SPAN/GPGGA truth axis contains severe jumps.",
        "- `9_seg01` has good median error but larger final endpoint error, showing why endpoint and coverage metrics are needed.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_df = evaluate()
    summary = summarize(eval_df)
    eval_df.to_csv(OUT_DIR / "endpoint_error_by_segment.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "endpoint_error_summary.csv", index=False, encoding="utf-8-sig")
    p1 = plot_final_errors(eval_df)
    p2 = plot_coverage(eval_df)
    write_notes(eval_df, summary, OUT_DIR / "endpoint_error_notes.md")
    print(summary.round(3).to_string(index=False))
    print()
    print(eval_df[eval_df["method"] == "IMUProgressClosest_TotalVsAxis"].round(3).to_string(index=False))
    print(p1)
    print(p2)


if __name__ == "__main__":
    main()
