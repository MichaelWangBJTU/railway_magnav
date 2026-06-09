from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import axis_calibrated_hmm_experiment as hmm


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\truth_axis_anomaly_diagnostic")
AXIS_VARIANT = "fwd_z_y_x_back_z_y_minusx"
SAMPLE_PERIOD = "4s"
BEST_RESULT_CSV = Path(r"C:\Users\m1352\Documents\railway_magnav\forward_anchor_hmm_tuning\forward_anchor_hmm_tuning_results.csv")
BEST_METHOD = "TotalHP_vmax1.2_uniform"


def diagnose_queries() -> pd.DataFrame:
    rows = []
    for q in hmm.read_query_segments(AXIS_VARIANT, SAMPLE_PERIOD):
        s = np.asarray(q.truth_s, dtype=float)
        ds = np.diff(s)
        expected = 1.0 if q.direction == "forward" else -1.0
        sign_violation = int(((expected * ds) < -0.5).sum())
        large_jump = int((np.abs(ds) > 20.0).sum())
        max_abs_jump = float(np.nanmax(np.abs(ds))) if len(ds) else 0.0
        median_abs_step = float(np.nanmedian(np.abs(ds))) if len(ds) else 0.0
        warning = int(large_jump > 0 or sign_violation > max(3, 0.05 * len(ds)))
        severe = int(max_abs_jump > 100.0 or large_jump >= 3 or sign_violation > max(30, 0.15 * len(ds)))
        rows.append(
            {
                "segment_label": q.label,
                "direction": q.direction,
                "sample_count": int(len(s)),
                "s_start_m": float(s[0]),
                "s_end_m": float(s[-1]),
                "s_min_m": float(np.nanmin(s)),
                "s_max_m": float(np.nanmax(s)),
                "sign_violation_count": sign_violation,
                "large_jump_count_gt20m": large_jump,
                "max_abs_step_m": max_abs_jump,
                "median_abs_step_m": median_abs_step,
                "truth_axis_warning": warning,
                "severe_truth_axis_anomaly": severe,
            }
        )
    return pd.DataFrame(rows)


def summarize_best_with_truth_flags(diag: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not BEST_RESULT_CSV.exists():
        return pd.DataFrame(), pd.DataFrame()
    results = pd.read_csv(BEST_RESULT_CSV)
    best = results[results["method"] == BEST_METHOD].copy()
    best = best.merge(diag[["segment_label", "truth_axis_warning", "severe_truth_axis_anomaly"]], on="segment_label", how="left")
    rows = []
    for label, g in [
        ("all_usable_segments", best),
        ("exclude_severe_truth_axis", best[best["severe_truth_axis_anomaly"] == 0]),
        ("strict_no_warning_only", best[best["truth_axis_warning"] == 0]),
    ]:
        if g.empty:
            continue
        rows.append(
            {
                "evaluation_set": label,
                "segment_count": int(len(g)),
                "median_abs_error_m": float(g["median_abs_error_m"].median()),
                "mean_abs_error_m": float(g["mean_abs_error_m"].mean()),
                "rmse_m": float(g["rmse_m"].mean()),
                "p90_abs_error_m": float(g["p90_abs_error_m"].mean()),
                "final_abs_error_m": float(g["final_abs_error_m"].median()),
            }
        )
    return best, pd.DataFrame(rows)


def write_notes(diag: pd.DataFrame, best: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Truth-axis Anomaly Diagnostic",
        "",
        "Purpose: check whether poor localization segments are caused by magnetic mismatch or by non-physical SPAN/GPGGA projected-distance truth.",
        "",
        "Rules:",
        "",
        "- A segment is flagged if the resampled SPAN distance contains any step larger than 20 m between consecutive 4 s samples.",
        "- It is also flagged if many steps move against the declared travel direction.",
        "- This diagnostic is only for evaluation integrity; the HMM itself does not use truth position.",
        "",
        "Truth-axis diagnostics:",
        "",
        diag.to_markdown(index=False, floatfmt=".3f"),
        "",
        f"Best current method: `{BEST_METHOD}`",
        "",
        "Best-method segment errors with truth-axis flags:",
        "",
        best[
            [
                "segment_label",
                "direction",
                "truth_axis_warning",
                "severe_truth_axis_anomaly",
                "median_abs_error_m",
                "mean_abs_error_m",
                "rmse_m",
                "p90_abs_error_m",
                "final_abs_error_m",
            ]
        ].to_markdown(index=False, floatfmt=".3f") if not best.empty else "(best result file not found)",
        "",
        "Summary with and without flagged truth-axis segments:",
        "",
        summary.to_markdown(index=False, floatfmt=".3f") if not summary.empty else "(summary unavailable)",
        "",
        "Interpretation:",
        "",
        "- `BMAW15230010L_1_seg03` is a severe truth-axis anomaly: the projected SPAN distance has multiple jumps above 190 m and even changes direction.",
        "- `BMAW15230010L_1_seg01` has mild warnings but the HMM remains accurate on it, so the stricter severe flag is a better exclusion rule than any-jump filtering.",
        "- The magnetic curve of this segment can still align to the map under bounded DTW, so it should not be treated as a simple magnetic failure.",
        "- The main reported no-wheel localization metric should include the all-segment result for honesty, and also report an exclude-severe-truth-axis metric for method capability.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    diag = diagnose_queries()
    best, summary = summarize_best_with_truth_flags(diag)
    diag.to_csv(OUT_DIR / "truth_axis_anomaly_by_segment.csv", index=False, encoding="utf-8-sig")
    best.to_csv(OUT_DIR / "best_method_errors_with_truth_flags.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "best_method_truth_clean_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "truth_axis_anomaly_summary.json").write_text(
        json.dumps(
            {
                "diagnostic": diag.to_dict(orient="records"),
                "best_method_summary": summary.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_notes(diag, best, summary, OUT_DIR / "truth_axis_anomaly_notes.md")
    print(diag.to_string(index=False))
    print()
    print(summary.round(3).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
