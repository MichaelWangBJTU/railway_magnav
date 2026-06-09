from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\axis_calibrated_experiment")


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_results() -> pd.DataFrame:
    paths = [
        OUT_DIR / "matching_results_all_axis_variants.csv",
        OUT_DIR / "quality_good_matching_results_all_axis_variants.csv",
    ]
    frames = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "reference_profile" not in df.columns:
            df["reference_profile"] = "all"
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No matching result CSV files found")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["prior_radius_m"] == "global"].copy()
    df = df.dropna(subset=["abs_error_m", "best_score", "score_gap"])
    df["is_good_5m"] = df["abs_error_m"] <= 5.0
    df["is_good_10m"] = df["abs_error_m"] <= 10.0
    df["is_good_20m"] = df["abs_error_m"] <= 20.0
    return df


def summarize_accept(g: pd.DataFrame, accepted: pd.DataFrame, threshold: float, threshold_kind: str) -> dict[str, object]:
    if accepted.empty:
        return {
            "threshold_kind": threshold_kind,
            "threshold": threshold,
            "accepted_count": 0,
            "accept_rate": 0.0,
            "median_abs_error_m": np.nan,
            "p75_abs_error_m": np.nan,
            "p90_abs_error_m": np.nan,
            "rmse_m": np.nan,
            "good_rate_5m": np.nan,
            "good_rate_10m": np.nan,
            "good_rate_20m": np.nan,
        }
    err = accepted["error_m"].to_numpy(float)
    return {
        "threshold_kind": threshold_kind,
        "threshold": float(threshold),
        "accepted_count": int(len(accepted)),
        "accept_rate": float(len(accepted) / len(g)),
        "median_abs_error_m": float(accepted["abs_error_m"].median()),
        "p75_abs_error_m": float(accepted["abs_error_m"].quantile(0.75)),
        "p90_abs_error_m": float(accepted["abs_error_m"].quantile(0.90)),
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "good_rate_5m": float(accepted["is_good_5m"].mean()),
        "good_rate_10m": float(accepted["is_good_10m"].mean()),
        "good_rate_20m": float(accepted["is_good_20m"].mean()),
    }


def gate_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    focus = df[
        df["method"].isin(["total_raw_hp_ncc", "axiscal_xy_hp_msd", "axiscal_xy_hp_ncc", "axiscal_xy_total_hp_ncc"])
        & df["window_m"].isin([50.0, 100.0, 150.0])
    ].copy()
    groups = ["reference_profile", "axis_variant", "method", "window_m"]
    rows = []
    per_segment_rows = []
    for key, g in focus.groupby(groups, dropna=False):
        base = dict(zip(groups, key))
        g = g.sort_values("score_gap")
        abs_thresholds = np.unique(
            np.r_[
                np.linspace(0.00, 0.30, 16),
                np.nanquantile(g["score_gap"], [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]),
            ]
        )
        for thr in abs_thresholds:
            accepted = g[g["score_gap"] >= thr]
            row = base.copy()
            row.update(summarize_accept(g, accepted, float(thr), "score_gap_abs"))
            rows.append(row)
        for accept_rate in [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10]:
            thr = float(g["score_gap"].quantile(1.0 - accept_rate))
            accepted = g[g["score_gap"] >= thr]
            row = base.copy()
            row.update(summarize_accept(g, accepted, thr, f"top_{int(accept_rate * 100)}pct_by_gap"))
            rows.append(row)

        for seg, sg in g.groupby("query_segment"):
            top_thr = float(g["score_gap"].quantile(0.80))
            acc = sg[sg["score_gap"] >= top_thr]
            per_segment_rows.append(
                {
                    **base,
                    "query_segment": seg,
                    "query_direction": sg["query_direction"].iloc[0],
                    "segment_count": int(len(sg)),
                    "segment_median_abs_error_all_m": float(sg["abs_error_m"].median()),
                    "accepted_count_at_global_q80": int(len(acc)),
                    "accept_rate_at_global_q80": float(len(acc) / len(sg)) if len(sg) else np.nan,
                    "median_abs_error_at_global_q80_m": float(acc["abs_error_m"].median()) if len(acc) else np.nan,
                    "p75_abs_error_at_global_q80_m": float(acc["abs_error_m"].quantile(0.75)) if len(acc) else np.nan,
                }
            )
    gate = pd.DataFrame(rows)
    per_seg = pd.DataFrame(per_segment_rows)
    return gate, per_seg


def best_gate_summary(gate: pd.DataFrame) -> pd.DataFrame:
    # Keep practically interesting rows: not tiny coverage, p75 below a useful bound.
    filtered = gate[
        (gate["threshold_kind"].str.startswith("top_"))
        & (gate["accept_rate"] >= 0.15)
        & (gate["p75_abs_error_m"].notna())
    ].copy()
    filtered["rank"] = (
        filtered["median_abs_error_m"]
        + 0.5 * filtered["p75_abs_error_m"]
        + 20.0 * (0.30 - filtered["accept_rate"]).clip(lower=0)
    )
    return filtered.sort_values(["rank", "accept_rate"], ascending=[True, False])


def plot_tradeoffs(gate: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    focus = gate[
        (gate["threshold_kind"].str.startswith("top_"))
        & (gate["reference_profile"].isin(["all", "quality_good"]))
    ].copy()
    for method in ["total_raw_hp_ncc", "axiscal_xy_hp_msd", "axiscal_xy_total_hp_ncc"]:
        for window in [100.0, 150.0]:
            sub = focus[(focus["method"] == method) & (focus["window_m"] == window)]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
            for (ref_profile, variant), g in sub.groupby(["reference_profile", "axis_variant"], dropna=False):
                g = g.sort_values("accept_rate")
                label = f"{ref_profile}/{variant[:12]}"
                ax.plot(g["accept_rate"], g["median_abs_error_m"], marker="o", label=label)
            ax.set_title(f"Uniqueness gate tradeoff: {method}, {window:.0f} m window")
            ax.set_xlabel("Accepted fraction")
            ax.set_ylabel("Median abs error of accepted windows / m")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = OUT_DIR / f"uniqueness_tradeoff_{method}_{int(window)}m.png"
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)
    return paths


def write_notes(best: pd.DataFrame, per_seg: pd.DataFrame, paths: list[Path]) -> None:
    note = OUT_DIR / "uniqueness_gate_notes.md"
    top = best.head(30).copy()
    seg_focus = per_seg[
        (per_seg["method"] == "total_raw_hp_ncc")
        & (per_seg["window_m"].isin([100.0, 150.0]))
        & (per_seg["reference_profile"] == "all")
    ].copy()
    lines = [
        "# Uniqueness Gate Analysis",
        "",
        "This analysis uses only global matching score curves. The gate is therefore a real matching-side confidence measure, unlike the prior-limited experiments that use the true position to emulate a search window.",
        "",
        "The key variable is `score_gap = best_score - second_score`, where the second score is the best candidate at least 10 m away from the best candidate.",
        "",
        "## Best Accepted-Set Rows",
        "",
        top.to_markdown(index=False),
        "",
        "## Per-Segment Check for Total-Field High-Pass NCC",
        "",
        seg_focus.sort_values(["window_m", "query_segment"]).to_markdown(index=False),
        "",
        "## Figures",
        "",
    ]
    for p in paths:
        lines.append(f"- `{p}`")
    note.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_matplotlib()
    df = load_results()
    gate, per_seg = gate_tables(df)
    gate_path = OUT_DIR / "uniqueness_gate_summary.csv"
    per_seg_path = OUT_DIR / "uniqueness_gate_per_segment.csv"
    gate.to_csv(gate_path, index=False, encoding="utf-8-sig")
    per_seg.to_csv(per_seg_path, index=False, encoding="utf-8-sig")
    best = best_gate_summary(gate)
    best_path = OUT_DIR / "uniqueness_gate_best_rows.csv"
    best.to_csv(best_path, index=False, encoding="utf-8-sig")
    paths = plot_tradeoffs(gate)
    write_notes(best, per_seg, paths)
    cols = [
        "reference_profile",
        "axis_variant",
        "method",
        "window_m",
        "threshold_kind",
        "threshold",
        "accept_rate",
        "median_abs_error_m",
        "p75_abs_error_m",
        "rmse_m",
        "good_rate_5m",
        "good_rate_10m",
        "good_rate_20m",
    ]
    print(best[cols].head(40).round(3).to_string(index=False))
    print(f"\nSaved: {gate_path}\n{per_seg_path}\n{best_path}")


if __name__ == "__main__":
    main()
