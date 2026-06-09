from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import axis_calibrated_full_matching as ac
import distance_warp_diagnostic as dwd


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\constrained_map_alignment_experiment")
AXIS_VARIANT = "fwd_z_y_x_back_z_y_minusx"
FEATURE_NAMES = ["total_hp", "axis_x_hp", "axis_y_hp", "axis_z_hp", "axis_total_hp"]
ALIGN_FEATURE = "total_hp"
BAND_M = 60.0


@dataclass
class PassData:
    date_tag: str
    segment: str
    direction: str
    distance: np.ndarray
    features: dict[str, np.ndarray]


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_passes(date_tag: str) -> tuple[np.ndarray, list[PassData]]:
    df = ac.load_map(date_tag)
    segs = ac.load_segments(date_tag)
    dist = pd.to_numeric(df["distance_m"], errors="coerce").to_numpy(float)
    passes: list[PassData] = []
    for _, row in segs.iterrows():
        seg = str(row["segment_label"])
        direction = str(row["direction"])
        feats = ac.segment_track_features(df, seg, date_tag, direction, AXIS_VARIANT)
        features = {
            "total_hp": feats["total_hp"],
            "axis_x_hp": feats["track_x_hp"],
            "axis_y_hp": feats["track_y_hp"],
            "axis_z_hp": feats["track_z_hp"],
            "axis_total_hp": feats["total_hp"],
        }
        passes.append(PassData(date_tag, seg, direction, dist, features))
    return dist, passes


def interpolate_clean(x: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype=float)).interpolate(limit_direction="both").to_numpy(float)


def robust_stack(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.array([])
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(np.vstack(arrays), axis=0)
    return interpolate_clean(med)


def build_reference_from_arrays(dist: np.ndarray, aligned_arrays: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    ref: dict[str, np.ndarray] = {"distance_m": dist}
    for feat in FEATURE_NAMES:
        ref[feat] = robust_stack([a[feat] for a in aligned_arrays if feat in a])
        ref[feat + "_z"] = dwd.z_valid(ref[feat])
    return ref


def raw_reference(dist: np.ndarray, passes: list[PassData]) -> dict[str, np.ndarray]:
    arrays = [{feat: p.features[feat] for feat in FEATURE_NAMES} for p in passes]
    return build_reference_from_arrays(dist, arrays)


def mapping_to_reference(
    p: PassData,
    ref: dict[str, np.ndarray],
    band_m: float = BAND_M,
    downsample_step: int = 2,
) -> tuple[np.ndarray, dict[str, float]]:
    q_s = p.distance
    q_feat = interpolate_clean(p.features[ALIGN_FEATURE])
    ref_s = ref["distance_m"]
    ref_feat = interpolate_clean(ref[ALIGN_FEATURE])

    mask = np.isfinite(q_feat)
    q_s_valid = q_s[mask]
    q_z = dwd.z_valid(q_feat[mask])
    q_s_ds = q_s_valid[::downsample_step]
    q_z_ds = q_z[::downsample_step]
    ref_s_ds = ref_s[::downsample_step]
    ref_z_ds = dwd.z_valid(ref_feat)[::downsample_step]

    _, rj, cost = dwd.distance_banded_dtw(q_s_ds, q_z_ds, ref_s_ds, ref_z_ds, band_m=band_m)
    valid = rj >= 0
    mapped_ds = ref_s_ds[np.clip(rj, 0, len(ref_s_ds) - 1)]
    mapped_ds[~valid] = q_s_ds[~valid]
    mapped_ds = np.maximum.accumulate(mapped_ds)
    mapped_full_valid = np.interp(q_s_valid, q_s_ds, mapped_ds, left=mapped_ds[0], right=mapped_ds[-1])
    mapped_full = np.full_like(q_s, np.nan, dtype=float)
    mapped_full[mask] = mapped_full_valid
    mapped_full = pd.Series(mapped_full).interpolate(limit_direction="both").to_numpy(float)
    mapped_full = np.maximum.accumulate(mapped_full)

    aligned_ref_z = np.interp(mapped_ds, ref_s_ds, ref_z_ds, left=np.nan, right=np.nan)
    corr = float(np.corrcoef(q_z_ds[valid], aligned_ref_z[valid])[0, 1]) if valid.sum() > 2 else math.nan
    axis_diff = np.abs(mapped_ds[valid] - q_s_ds[valid]) if valid.any() else np.array([np.nan])
    return mapped_full, {
        "align_corr": corr,
        "align_cost": float(cost),
        "axis_median_abs_diff_m": float(np.nanmedian(axis_diff)),
        "axis_p90_abs_diff_m": float(np.nanpercentile(axis_diff, 90)),
        "valid_rate": float(valid.mean()),
    }


def resample_pass_to_ref(p: PassData, mapped_s: np.ndarray, ref_s: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(mapped_s)
    x = mapped_s[order]
    # Collapse repeated DTW coordinates, otherwise interpolation can jitter.
    rounded = np.round(x * 2.0) / 2.0
    out: dict[str, np.ndarray] = {}
    for feat in FEATURE_NAMES:
        y = interpolate_clean(p.features[feat])[order]
        tmp = pd.DataFrame({"x": rounded, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
        if tmp.empty:
            out[feat] = np.full_like(ref_s, np.nan, dtype=float)
            continue
        grouped = tmp.groupby("x", as_index=False)["y"].median().sort_values("x")
        gx = grouped["x"].to_numpy(float)
        gy = grouped["y"].to_numpy(float)
        if len(gx) < 2:
            out[feat] = np.full_like(ref_s, np.nan, dtype=float)
        else:
            out[feat] = np.interp(ref_s, gx, gy, left=np.nan, right=np.nan)
    return out


def aligned_reference(
    dist: np.ndarray,
    passes: list[PassData],
    base_ref: dict[str, np.ndarray] | None = None,
    iterations: int = 2,
    exclude_segment: str | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, list[dict[str, np.ndarray]]]:
    use_passes = [p for p in passes if p.segment != exclude_segment]
    ref = raw_reference(dist, use_passes) if base_ref is None else base_ref
    all_align_rows = []
    aligned_arrays: list[dict[str, np.ndarray]] = []
    for it in range(iterations):
        aligned_arrays = []
        align_rows = []
        for p in use_passes:
            mapped, meta = mapping_to_reference(p, ref, BAND_M)
            aligned = resample_pass_to_ref(p, mapped, dist)
            aligned_arrays.append(aligned)
            align_rows.append(
                {
                    "iteration": it + 1,
                    "date_tag": p.date_tag,
                    "segment_label": p.segment,
                    "direction": p.direction,
                    **meta,
                }
            )
        ref = build_reference_from_arrays(dist, aligned_arrays)
        all_align_rows.extend(align_rows)
    return ref, pd.DataFrame(all_align_rows), aligned_arrays


def selective_aligned_reference(
    dist: np.ndarray,
    passes: list[PassData],
    corr_threshold: float = 0.82,
    p90_threshold_m: float = 55.0,
    iterations: int = 1,
    exclude_segment: str | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, list[dict[str, np.ndarray]]]:
    use_passes = [p for p in passes if p.segment != exclude_segment]
    ref = raw_reference(dist, use_passes)
    all_rows = []
    aligned_arrays: list[dict[str, np.ndarray]] = []
    for it in range(iterations):
        aligned_arrays = []
        rows = []
        for p in use_passes:
            mapped, meta = mapping_to_reference(p, ref, BAND_M)
            accept = bool(
                np.isfinite(meta["align_corr"])
                and meta["align_corr"] >= corr_threshold
                and meta["axis_p90_abs_diff_m"] <= p90_threshold_m
            )
            if not accept:
                mapped = p.distance.copy()
            aligned = resample_pass_to_ref(p, mapped, dist)
            aligned_arrays.append(aligned)
            rows.append(
                {
                    "iteration": it + 1,
                    "date_tag": p.date_tag,
                    "segment_label": p.segment,
                    "direction": p.direction,
                    "accepted_alignment": int(accept),
                    **meta,
                }
            )
        ref = build_reference_from_arrays(dist, aligned_arrays)
        all_rows.extend(rows)
    return ref, pd.DataFrame(all_rows), aligned_arrays


def map_spread(arrays: list[dict[str, np.ndarray]], feat: str = ALIGN_FEATURE) -> dict[str, float]:
    stack = []
    for a in arrays:
        z = dwd.z_valid(a[feat])
        stack.append(z)
    mat = np.vstack(stack)
    std = np.nanstd(mat, axis=0)
    return {
        "spread_median_std_z": float(np.nanmedian(std)),
        "spread_p75_std_z": float(np.nanpercentile(std, 75)),
        "spread_p90_std_z": float(np.nanpercentile(std, 90)),
    }


def eval_against_ref(p: PassData, ref: dict[str, np.ndarray], label: str) -> dict[str, float | str]:
    q_s = p.distance
    q_z = dwd.z_valid(p.features[ALIGN_FEATURE])
    ref_s = ref["distance_m"]
    ref_z = dwd.z_valid(ref[ALIGN_FEATURE])
    ident = dwd.identity_metrics(q_s, q_z, ref_s, ref_z)
    _, rj, cost = dwd.distance_banded_dtw(q_s[::2], q_z[::2], ref_s[::2], ref_z[::2], BAND_M)
    valid = rj >= 0
    mapped = ref_s[::2][np.clip(rj, 0, len(ref_s[::2]) - 1)]
    aligned_z = ref_z[::2][np.clip(rj, 0, len(ref_s[::2]) - 1)]
    q_z_ds = q_z[::2]
    bcorr = float(np.corrcoef(q_z_ds[valid], aligned_z[valid])[0, 1]) if valid.sum() > 2 else math.nan
    brms = float(np.sqrt(np.mean((q_z_ds[valid] - aligned_z[valid]) ** 2))) if valid.any() else math.nan
    axis_diff = np.abs(mapped[valid] - q_s[::2][valid]) if valid.any() else np.array([np.nan])
    return {
        "eval_ref": label,
        "segment_label": p.segment,
        "date_tag": p.date_tag,
        "direction": p.direction,
        **ident,
        "band60_dtw_corr": bcorr,
        "band60_dtw_rms_z": brms,
        "band60_dtw_cost": float(cost),
        "band60_axis_median_abs_diff_m": float(np.nanmedian(axis_diff)),
        "band60_axis_p90_abs_diff_m": float(np.nanpercentile(axis_diff, 90)),
    }


def evaluate_lopo(dist: np.ndarray, passes: list[PassData]) -> pd.DataFrame:
    rows = []
    for p in passes:
        others = [x for x in passes if x.segment != p.segment]
        raw_ref = raw_reference(dist, others)
        corrected_ref, _, _ = aligned_reference(dist, passes, iterations=2, exclude_segment=p.segment)
        selective_ref, _, _ = selective_aligned_reference(dist, passes, iterations=1, exclude_segment=p.segment)
        rows.append(eval_against_ref(p, raw_ref, "raw_lopo_ref"))
        rows.append(eval_against_ref(p, corrected_ref, "corrected_lopo_ref"))
        rows.append(eval_against_ref(p, selective_ref, "selective_lopo_ref"))
    return pd.DataFrame(rows)


def evaluate_cross_day(
    passes_5: list[PassData],
    raw_ref: dict[str, np.ndarray],
    corrected_ref: dict[str, np.ndarray],
    selective_ref: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for p in passes_5:
        rows.append(eval_against_ref(p, raw_ref, "raw_4_14_ref"))
        rows.append(eval_against_ref(p, corrected_ref, "corrected_4_14_ref"))
        rows.append(eval_against_ref(p, selective_ref, "selective_4_14_ref"))
    return pd.DataFrame(rows)


def summarize_eval(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in eval_df.groupby("eval_ref"):
        rows.append(
            {
                "eval_ref": label,
                "segment_count": int(len(g)),
                "median_identity_corr": float(g["identity_corr"].median()),
                "mean_identity_corr": float(g["identity_corr"].mean()),
                "median_band60_corr": float(g["band60_dtw_corr"].median()),
                "mean_band60_corr": float(g["band60_dtw_corr"].mean()),
                "median_band60_rms_z": float(g["band60_dtw_rms_z"].median()),
                "median_axis_correction_m": float(g["band60_axis_median_abs_diff_m"].median()),
                "p90_axis_correction_m": float(g["band60_axis_p90_abs_diff_m"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("eval_ref")


def plot_reference(dist: np.ndarray, raw_ref: dict[str, np.ndarray], corr_ref: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=180)
    ax.plot(dist, dwd.z_valid(raw_ref[ALIGN_FEATURE]), label="Raw 4.14 reference", lw=1.2)
    ax.plot(dist, dwd.z_valid(corr_ref[ALIGN_FEATURE]), label="Constrained-aligned 4.14 reference", lw=1.2)
    ax.set_title("Reference map before/after constrained pass alignment")
    ax.set_xlabel("Reference distance / m")
    ax.set_ylabel("Total HP robust z")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=180)
    labels = summary["eval_ref"].tolist()
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, summary["median_identity_corr"], width, label="identity corr")
    ax.bar(x + width / 2, summary["median_band60_corr"], width, label="banded DTW corr")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(-0.1, 1.0)
    ax.set_ylabel("Median correlation")
    ax.set_title("Map consistency before/after constrained alignment")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_notes(spread_df: pd.DataFrame, eval_summary: pd.DataFrame, align_df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Constrained Map Alignment Experiment",
        "",
        "Purpose: build a 4.14 reference map after constrained per-pass distance-axis alignment, then test same-day LOPO and 5.13 cross-day consistency.",
        "",
        "Important boundary:",
        "",
        "- This is an offline map-construction experiment. Full-pass banded DTW is allowed for building a magnetic map, but not for online real-time localization.",
        "- The alignment feature is total-field high-pass (`total_hp`) and the distance correction is constrained within +/-60 m around the original projected distance axis.",
        "",
        "Map spread:",
        "",
        spread_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Evaluation summary:",
        "",
        eval_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Alignment diagnostics, final iteration:",
        "",
        align_df[align_df["iteration"] == align_df["iteration"].max()].to_markdown(index=False, floatfmt=".3f"),
        "",
        "Interpretation guide:",
        "",
        "- If corrected LOPO spread decreases and LOPO/cross-day correlation improves, constrained distance-axis self-calibration is a valid map-construction module.",
        "- If identity correlation does not improve but banded DTW correlation remains high, the corrected map helps offline consistency but has not yet solved online localization.",
        "- If alignment corrections cluster near the +/-60 m band edge, the band may be too tight or the initial distance axis may contain larger systematic errors.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dist4, passes4 = load_passes("4_14")
    _, passes5 = load_passes("5_13")
    raw_ref4 = raw_reference(dist4, passes4)
    raw_arrays4 = [{feat: p.features[feat] for feat in FEATURE_NAMES} for p in passes4]
    corr_ref4, align_df, corr_arrays4 = aligned_reference(dist4, passes4, iterations=2)
    sel_ref4, sel_align_df, sel_arrays4 = selective_aligned_reference(dist4, passes4, iterations=1)

    spread_rows = [
        {"map": "raw_4_14", **map_spread(raw_arrays4)},
        {"map": "corrected_4_14", **map_spread(corr_arrays4)},
        {"map": "selective_4_14", **map_spread(sel_arrays4)},
    ]
    spread_df = pd.DataFrame(spread_rows)
    lopo_eval = evaluate_lopo(dist4, passes4)
    cross_eval = evaluate_cross_day(passes5, raw_ref4, corr_ref4, sel_ref4)
    eval_df = pd.concat([lopo_eval, cross_eval], ignore_index=True)
    eval_summary = summarize_eval(eval_df)

    # Export corrected map.
    corrected_map = pd.DataFrame({"distance_m": dist4})
    for feat in FEATURE_NAMES:
        corrected_map[f"corrected_{feat}"] = corr_ref4[feat]
        corrected_map[f"selective_{feat}"] = sel_ref4[feat]
        corrected_map[f"raw_{feat}"] = raw_ref4[feat]
    corrected_map.to_csv(OUT_DIR / "corrected_4_14_reference_map.csv", index=False, encoding="utf-8-sig")
    align_df.to_csv(OUT_DIR / "alignment_diagnostics_4_14.csv", index=False, encoding="utf-8-sig")
    sel_align_df.to_csv(OUT_DIR / "selective_alignment_diagnostics_4_14.csv", index=False, encoding="utf-8-sig")
    spread_df.to_csv(OUT_DIR / "map_spread_summary.csv", index=False, encoding="utf-8-sig")
    eval_df.to_csv(OUT_DIR / "alignment_evaluation_by_segment.csv", index=False, encoding="utf-8-sig")
    eval_summary.to_csv(OUT_DIR / "alignment_evaluation_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "constrained_map_alignment_summary.json").write_text(
        json.dumps(
            {
                "spread": spread_df.to_dict(orient="records"),
                "eval_summary": eval_summary.to_dict(orient="records"),
                "alignment": align_df.to_dict(orient="records"),
                "selective_alignment": sel_align_df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_reference(dist4, raw_ref4, corr_ref4, OUT_DIR / "corrected_reference_compare.png")
    plot_reference(dist4, raw_ref4, sel_ref4, OUT_DIR / "selective_reference_compare.png")
    plot_summary(eval_summary, OUT_DIR / "alignment_evaluation_summary.png")
    write_notes(spread_df, eval_summary, align_df, OUT_DIR / "constrained_map_alignment_notes.md")
    print("Map spread:")
    print(spread_df.round(3).to_string(index=False))
    print("\nEvaluation summary:")
    print(eval_summary.round(3).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
