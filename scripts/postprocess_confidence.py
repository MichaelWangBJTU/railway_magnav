from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path.home() / "Desktop" / "磁导航" / "数据" / "codex_railway_magnav"
OUT_ROOT = PROJECT_ROOT / "sota_repro"


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_results(out_root: Path) -> pd.DataFrame:
    outputs = out_root / "outputs"
    frames = []
    paths = [
        outputs / "distinctive_subsequence_results.csv",
        outputs / "weak_mileage_alignment_results.csv",
        outputs / "baseline_matching_window_results.csv",
    ]
    for p in paths:
        if p.exists():
            df = pd.read_csv(p)
            if "score_margin" in df.columns and "abs_error_m" in df.columns:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[np.isfinite(df["score_margin"])].copy()


def summarize_confidence(df: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    rows = []
    thresholds = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    keep_methods = [
        "DistinctiveSubseq_highpass_grad",
        "WeakMileage_highpass_grad",
        "WeakMileage_total_y_highpass",
        "NCC_total_grad",
    ]
    df = df[df["method"].isin(keep_methods)].copy()
    for method, g in df.groupby("method"):
        for th in thresholds:
            sub = g[g["score_margin"] >= th]
            if sub.empty:
                rows.append(
                    {
                        "method": method,
                        "margin_threshold": th,
                        "accepted_count": 0,
                        "total_count": len(g),
                        "coverage_pct": 0.0,
                        "median_abs_error_m": np.nan,
                        "mean_abs_error_m": np.nan,
                        "p90_abs_error_m": np.nan,
                    }
                )
            else:
                rows.append(
                    {
                        "method": method,
                        "margin_threshold": th,
                        "accepted_count": len(sub),
                        "total_count": len(g),
                        "coverage_pct": 100.0 * len(sub) / len(g),
                        "median_abs_error_m": float(sub["abs_error_m"].median()),
                        "mean_abs_error_m": float(sub["abs_error_m"].mean()),
                        "p90_abs_error_m": float(np.nanpercentile(sub["abs_error_m"], 90)),
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "outputs" / "confidence_gating_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def plot(summary: pd.DataFrame, out_root: Path) -> None:
    if summary.empty:
        return
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=160)
    ax2 = ax1.twinx()
    for method, g in summary.groupby("method"):
        ax1.plot(g["margin_threshold"], g["median_abs_error_m"], marker="o", lw=1.6, label=f"{method} 误差")
        ax2.plot(g["margin_threshold"], g["coverage_pct"], ls="--", lw=1.2, alpha=0.7, label=f"{method} 覆盖率")
    ax1.set_xlabel("唯一性置信度门限：best score - second score")
    ax1.set_ylabel("接受样本中位绝对误差 / m")
    ax2.set_ylabel("覆盖率 / %")
    ax1.set_title("置信度门控：用特征唯一性筛掉歧义匹配")
    ax1.grid(True, alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_root / "figures" / "confidence_gating_tradeoff.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    setup_matplotlib()
    df = load_results(args.out_root)
    summary = summarize_confidence(df, args.out_root)
    plot(summary, args.out_root)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
