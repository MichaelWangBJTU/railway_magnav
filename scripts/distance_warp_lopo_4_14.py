from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import axis_calibrated_full_matching as ac
import distance_warp_diagnostic as dwd


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\distance_warp_lopo_4_14")


def mean_stack(arrays: list[np.ndarray]) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.vstack(arrays), axis=0)


def run() -> None:
    dwd.setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = ac.load_map("4_14")
    segs = ac.load_segments("4_14")
    dist = df["distance_m"].to_numpy(float)
    rows = []
    for _, row in segs.iterrows():
        qseg = str(row["segment_label"])
        direction = str(row["direction"])
        q_feat = ac.segment_track_features(df, qseg, "4_14", direction)["total_hp"]
        ref_parts = []
        for _, r2 in segs.iterrows():
            seg = str(r2["segment_label"])
            if seg == qseg:
                continue
            ref_parts.append(ac.segment_track_features(df, seg, "4_14", str(r2["direction"]))["total_hp"])
        ref_feat = mean_stack(ref_parts)
        mask = np.isfinite(q_feat)
        if mask.sum() < 100:
            continue
        q_s = dist[mask]
        q_z = dwd.z_valid(q_feat[mask])
        ref_z = dwd.z_valid(ref_feat)
        step = max(1, int(round(len(q_s) / 500)))
        q_s_ds = q_s[::step]
        q_z_ds = q_z[::step]
        ref_step = max(1, int(round(len(dist) / 900)))
        ref_s_ds = dist[::ref_step]
        ref_z_ds = ref_z[::ref_step]
        ident = dwd.identity_metrics(q_s_ds, q_z_ds, ref_s_ds, ref_z_ds)
        _, rj, dtw_cost = dwd.subsequence_dtw(q_z_ds, ref_z_ds)
        _, brj, bdtw_cost = dwd.distance_banded_dtw(q_s_ds, q_z_ds, ref_s_ds, ref_z_ds, band_m=60.0)
        mapped_s = ref_s_ds[rj]
        aligned_z = ref_z_ds[rj]
        valid_b = brj >= 0
        bmapped_s = ref_s_ds[np.clip(brj, 0, len(ref_s_ds) - 1)]
        baligned_z = ref_z_ds[np.clip(brj, 0, len(ref_s_ds) - 1)]
        scale, offset, affine_s = dwd.fit_affine(q_s_ds, mapped_s)
        affine_res = mapped_s - affine_s
        abs_axis_err = np.abs(mapped_s - q_s_ds)
        b_abs_axis_err = np.abs(bmapped_s[valid_b] - q_s_ds[valid_b]) if valid_b.any() else np.asarray([np.nan])
        rows.append(
            {
                "segment_label": qseg,
                "direction": direction,
                "point_count": int(len(q_s_ds)),
                **ident,
                "dtw_corr": float(np.corrcoef(q_z_ds, aligned_z)[0, 1]),
                "dtw_rms_z": float(np.sqrt(np.mean((q_z_ds - aligned_z) ** 2))),
                "dtw_cost": float(dtw_cost),
                "band60_dtw_corr": float(np.corrcoef(q_z_ds[valid_b], baligned_z[valid_b])[0, 1]) if valid_b.sum() > 2 else np.nan,
                "band60_dtw_rms_z": float(np.sqrt(np.mean((q_z_ds[valid_b] - baligned_z[valid_b]) ** 2))) if valid_b.any() else np.nan,
                "band60_dtw_cost": float(bdtw_cost),
                "band60_axis_median_abs_diff_m": float(np.nanmedian(b_abs_axis_err)),
                "band60_axis_p90_abs_diff_m": float(np.nanpercentile(b_abs_axis_err, 90)),
                "dtw_axis_median_abs_diff_m": float(np.median(abs_axis_err)),
                "dtw_axis_p90_abs_diff_m": float(np.percentile(abs_axis_err, 90)),
                "affine_scale_ref_per_query": float(scale),
                "affine_offset_m": float(offset),
                "affine_residual_median_abs_m": float(np.median(np.abs(affine_res))),
                "affine_residual_p90_abs_m": float(np.percentile(np.abs(affine_res), 90)),
            }
        )
    summary = pd.DataFrame(rows).sort_values("identity_corr", ascending=False)
    summary.to_csv(OUT_DIR / "distance_warp_lopo_4_14_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "distance_warp_lopo_4_14_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 4.14 Leave-one-pass-out Distance Warp Diagnostic",
        "",
        "Purpose: check whether the distance-axis warp seen in 5.13 is a cross-day problem or already present in same-day 4.14 data.",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
    ]
    (OUT_DIR / "distance_warp_lopo_4_14_notes.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.round(3).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
