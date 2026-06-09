from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import msd_iccp_coarse_fine_check as coarse_fine


OUT_DIR = coarse_fine.OUT_DIR
AXES = ["x", "y", "z"]


def _series(date: str, segment: str, start: float, end: float):
    df = coarse_fine.load_map(date)
    return coarse_fine.segment_series(
        df,
        segment,
        coarse_fine.CHANNELS_BODY,
        start,
        end,
        step_m=1.0,
        smooth_points=3,
    )


def _same_grid_metrics(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, float]:
    row: dict[str, float] = {}
    for ch in ["x", "y", "z", "total"]:
        av = a[ch]
        bv = b[ch]
        mask = np.isfinite(av) & np.isfinite(bv)
        if mask.sum() < 10:
            row[f"{ch}_corr"] = np.nan
            row[f"{ch}_rms_nT"] = np.nan
            row[f"{ch}_rms_bias_removed_nT"] = np.nan
            row[f"{ch}_bias_nT"] = np.nan
            continue
        diff = bv[mask] - av[mask]
        bias = float(np.mean(diff))
        row[f"{ch}_corr"] = float(np.corrcoef(av[mask], bv[mask])[0, 1])
        row[f"{ch}_rms_nT"] = float(np.sqrt(np.mean(diff**2)))
        row[f"{ch}_rms_bias_removed_nT"] = float(np.sqrt(np.mean((diff - bias) ** 2)))
        row[f"{ch}_bias_nT"] = bias
    return row


def pairwise_true_distance_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    segs = {date: coarse_fine.load_segments(date) for date in ["4_14", "5_13"]}
    jobs = [
        ("same_day_4_14", "4_14", "4_14"),
        ("cross_day_4_14_5_13", "4_14", "5_13"),
    ]
    for tag, date_a, date_b in jobs:
        for _, a in segs[date_a].iterrows():
            for _, b in segs[date_b].iterrows():
                if date_a == date_b and a.segment_label >= b.segment_label:
                    continue
                if a.direction != b.direction:
                    continue
                start = max(75.0, float(a.s_min_m), float(b.s_min_m))
                end = min(265.0, float(a.s_max_m), float(b.s_max_m))
                if end - start < 80:
                    continue
                _, sa = _series(date_a, a.segment_label, start, end)
                _, sb = _series(date_b, b.segment_label, start, end)
                row: dict[str, object] = {
                    "tag": tag,
                    "date_a": date_a,
                    "date_b": date_b,
                    "segment_a": a.segment_label,
                    "segment_b": b.segment_label,
                    "direction": a.direction,
                    "start_m": start,
                    "end_m": end,
                    "length_m": end - start,
                }
                row.update(_same_grid_metrics(sa, sb))
                row["xy_mean_rms_bias_removed_nT"] = np.nanmean(
                    [row["x_rms_bias_removed_nT"], row["y_rms_bias_removed_nT"]]
                )
                row["mean_corr"] = np.nanmean(
                    [row["x_corr"], row["y_corr"], row["z_corr"], row["total_corr"]]
                )
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["tag", "xy_mean_rms_bias_removed_nT"])


def axis_permutation_table() -> pd.DataFrame:
    cases = [
        ("fwd_4_14_seg1_04_vs_5_13_seg1_04", "BMAW15230010L_1_seg04", "BMAW15230010L_1_seg04", 75.0, 265.0),
        ("fwd_4_14_seg5_01_vs_5_13_seg9_02", "BMAW15230010L_5_seg01", "BMAW15230010L_9_seg02", 87.5, 265.0),
        ("back_4_14_seg2_01_vs_5_13_seg1_03", "BMAW15230010L_2_seg01", "BMAW15230010L_1_seg03", 75.0, 256.0),
    ]
    rows: list[dict[str, object]] = []
    for case, ref_seg, query_seg, start, end in cases:
        _, ref = _series("4_14", ref_seg, start, end)
        _, query = _series("5_13", query_seg, start, end)
        for perm in itertools.permutations(AXES):
            for signs in itertools.product([-1, 1], repeat=3):
                mapped = {AXES[i]: signs[i] * query[perm[i]] for i in range(3)}
                rms_values = []
                corr_values = []
                bias_values = []
                ok = True
                for target in AXES:
                    a = ref[target]
                    b = mapped[target]
                    mask = np.isfinite(a) & np.isfinite(b)
                    if mask.sum() < 50:
                        ok = False
                        break
                    diff = b[mask] - a[mask]
                    bias = float(np.mean(diff))
                    rms_values.append(float(np.sqrt(np.mean((diff - bias) ** 2))))
                    corr_values.append(float(np.corrcoef(a[mask], b[mask])[0, 1]))
                    bias_values.append(bias)
                if not ok:
                    continue
                rows.append(
                    {
                        "case": case,
                        "ref_segment": ref_seg,
                        "query_segment": query_seg,
                        "map_x_from_5_13": f"{signs[0]:+d}{perm[0]}",
                        "map_y_from_5_13": f"{signs[1]:+d}{perm[1]}",
                        "map_z_from_5_13": f"{signs[2]:+d}{perm[2]}",
                        "mean_rms_bias_removed_nT": float(np.mean(rms_values)),
                        "x_rms_bias_removed_nT": rms_values[0],
                        "y_rms_bias_removed_nT": rms_values[1],
                        "z_rms_bias_removed_nT": rms_values[2],
                        "mean_corr": float(np.mean(corr_values)),
                        "x_corr": corr_values[0],
                        "y_corr": corr_values[1],
                        "z_corr": corr_values[2],
                        "x_bias_nT": bias_values[0],
                        "y_bias_nT": bias_values[1],
                        "z_bias_nT": bias_values[2],
                    }
                )
    return pd.DataFrame(rows).sort_values(["case", "mean_rms_bias_removed_nT"])


def plot_remapped_cross_day() -> Path:
    start, end = 87.5, 265.0
    ref_s, ref = _series("4_14", "BMAW15230010L_5_seg01", 0.0, 560.0)
    q_abs, q = _series("5_13", "BMAW15230010L_9_seg02", start, end)
    # The most physically plausible forward-day remounting correction is a
    # rotation about the down axis: 4.14-X ~= 5.13-Z, 4.14-Y ~= 5.13-Y,
    # 4.14-Z ~= -5.13-X. The vertical/right component still needs calibration.
    q_remap = {
        "x": q["z"],
        "y": q["y"],
        "z": -q["x"],
        "total": q["total"],
    }
    q_local = q_abs - q_abs[0]
    metrics, _ = coarse_fine.metrics_at_shift(
        q_local,
        q_remap,
        ref_s,
        ref,
        ["x", "y", "z", "total"],
        start,
    )
    out_path = OUT_DIR / "cross_day_axis_remapped_true_4_14_seg5_01_vs_5_13_seg9_02.png"
    coarse_fine.plot_case(
        out_path,
        "cross-day after 5.13 axis remap; true-distance alignment",
        q_local,
        q_remap,
        ref_s,
        ref,
        ["x", "y", "z", "total"],
        start,
        metrics,
    )
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairwise = pairwise_true_distance_table()
    pairwise_path = OUT_DIR / "pairwise_true_distance_body_window_75_265.csv"
    pairwise.to_csv(pairwise_path, index=False, encoding="utf-8-sig")

    axis_perm = axis_permutation_table()
    axis_path = OUT_DIR / "axis_permutation_cross_day_check.csv"
    axis_perm.to_csv(axis_path, index=False, encoding="utf-8-sig")

    remap_plot = plot_remapped_cross_day()
    print("Top same-day body-axis pairs:")
    print(pairwise[pairwise.tag == "same_day_4_14"].head(5).to_string(index=False))
    print("\nTop cross-day axis permutations:")
    print(axis_perm.groupby("case").head(3).to_string(index=False))
    print("\nSaved:")
    print(pairwise_path)
    print(axis_path)
    print(remap_plot)


if __name__ == "__main__":
    main()
