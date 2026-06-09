from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HMM_DIRS = {
    "default_4s": Path(r"C:\Users\m1352\Documents\railway_magnav\axis_calibrated_hmm_gate_sweep"),
    "sample_2s": Path(r"C:\Users\m1352\Documents\railway_magnav\axis_calibrated_hmm_2s"),
    "sample_6s": Path(r"C:\Users\m1352\Documents\railway_magnav\axis_calibrated_hmm_6s"),
    "sample_8s": Path(r"C:\Users\m1352\Documents\railway_magnav\axis_calibrated_hmm_8s"),
}
OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\axis_calibrated_hmm_diagnostics")


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    traj_frames = []
    result_frames = []
    for tag, root in HMM_DIRS.items():
        traj_path = root / "axis_calibrated_hmm_trajectories.csv"
        result_path = root / "axis_calibrated_hmm_results.csv"
        if traj_path.exists():
            t = pd.read_csv(traj_path)
            t["experiment"] = tag
            traj_frames.append(t)
        if result_path.exists():
            r = pd.read_csv(result_path)
            r["experiment"] = tag
            result_frames.append(r)
    return pd.concat(traj_frames, ignore_index=True), pd.concat(result_frames, ignore_index=True)


def plot_default_segments(traj: pd.DataFrame) -> list[Path]:
    paths = []
    focus_methods = [
        "Baseline_TotalHP_Viterbi",
        "AxisCal_XY_TotalHP_Viterbi",
        "AxisCal_XY_TotalHP_InfoGate_Viterbi",
        "AxisCal_XY_TotalHP_MidGate_Viterbi",
        "TotalHP_InfoGate_Viterbi",
    ]
    part = traj[(traj["experiment"] == "default_4s") & (traj["method"].isin(focus_methods))].copy()
    for seg, g in part.groupby("segment_label"):
        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=170)
        truth = g.groupby("time", sort=False)["truth_s_m"].first().reset_index(drop=True)
        ax.plot(np.arange(len(truth)), truth, color="black", lw=2.2, label="SPAN truth")
        for method in focus_methods:
            m = g[g["method"] == method].reset_index(drop=True)
            if m.empty:
                continue
            ax.plot(np.arange(len(m)), m["pred_s_m"], lw=1.2, label=method)
        ax.set_title(f"HMM trajectory diagnosis: {seg}")
        ax.set_xlabel("Resampled index")
        ax.set_ylabel("Along-track position / m")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = OUT_DIR / f"hmm_diagnosis_{seg}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def compare_sample_periods(results: pd.DataFrame) -> Path:
    methods = [
        "Baseline_TotalHP_Viterbi",
        "AxisCal_XY_TotalHP_InfoGate_Viterbi",
        "AxisCal_XY_TotalHP_MidGate_Viterbi",
        "AxisCal_XY_TotalHP_SoftGate_Viterbi",
        "AxisCal_XY_TotalHP_Viterbi",
    ]
    sub = results[results["method"].isin(methods)].copy()
    summary = (
        sub.groupby(["experiment", "method"])
        .agg(
            segment_count=("segment_label", "size"),
            median_abs_error_m=("median_abs_error_m", "median"),
            mean_abs_error_m=("mean_abs_error_m", "mean"),
            rmse_m=("rmse_m", "mean"),
            p90_abs_error_m=("p90_abs_error_m", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "hmm_sample_period_comparison.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=170)
    order = ["sample_2s", "default_4s", "sample_6s", "sample_8s"]
    for method, g in summary.groupby("method"):
        g = g.set_index("experiment").reindex(order).reset_index()
        ax.plot(order, g["median_abs_error_m"], marker="o", label=method)
    ax.set_title("HMM sample-period sensitivity")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Median abs error / m")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = OUT_DIR / "hmm_sample_period_sensitivity.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_notes(results: pd.DataFrame, paths: list[Path], sample_plot: Path) -> None:
    summary = (
        results.groupby(["experiment", "method"])
        .agg(
            segment_count=("segment_label", "size"),
            median_abs_error_m=("median_abs_error_m", "median"),
            mean_abs_error_m=("mean_abs_error_m", "mean"),
            rmse_m=("rmse_m", "mean"),
            final_abs_error_m=("final_abs_error_m", "median"),
        )
        .reset_index()
        .sort_values(["median_abs_error_m", "mean_abs_error_m"])
    )
    default_per_segment = results[
        (results["experiment"] == "default_4s")
        & results["method"].isin(
            [
                "Baseline_TotalHP_Viterbi",
                "AxisCal_XY_TotalHP_InfoGate_Viterbi",
                "AxisCal_XY_TotalHP_MidGate_Viterbi",
                "AxisCal_XY_TotalHP_SoftGate_Viterbi",
            ]
        )
    ].sort_values(["segment_label", "median_abs_error_m"])
    lines = [
        "# Axis-Calibrated HMM Diagnostics",
        "",
        "Main finding: the best median error comes from axis-calibrated X/Y + total high-pass features with an information gate at 4 s sampling, but two segments still fail badly. This supports treating the method as a confidence-aware localization system, not a blind always-on matcher.",
        "",
        "## Overall Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Default 4s Per-Segment Results",
        "",
        default_per_segment.to_markdown(index=False),
        "",
        "## Figures",
        "",
        f"- `{sample_plot}`",
    ]
    for p in paths:
        lines.append(f"- `{p}`")
    (OUT_DIR / "hmm_diagnostics_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    traj, results = load_all()
    paths = plot_default_segments(traj)
    sample_plot = compare_sample_periods(results)
    write_notes(results, paths, sample_plot)
    print(f"Saved diagnostics to: {OUT_DIR}")


if __name__ == "__main__":
    main()
