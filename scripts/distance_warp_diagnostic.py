from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import axis_calibrated_full_matching as ac


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\distance_warp_diagnostic")


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def z_valid(x: np.ndarray) -> np.ndarray:
    x = pd.Series(np.asarray(x, dtype=float)).interpolate(limit_direction="both").to_numpy(float)
    return ac.robust_z(x)


def identity_metrics(q_s: np.ndarray, q_z: np.ndarray, ref_s: np.ndarray, ref_z: np.ndarray) -> dict[str, float]:
    r = np.interp(q_s, ref_s, ref_z, left=np.nan, right=np.nan)
    mask = np.isfinite(q_z) & np.isfinite(r)
    if mask.sum() < 20:
        return {"identity_corr": math.nan, "identity_rms_z": math.nan}
    corr = float(np.corrcoef(q_z[mask], r[mask])[0, 1])
    rms = float(np.sqrt(np.mean((q_z[mask] - r[mask]) ** 2)))
    return {"identity_corr": corr, "identity_rms_z": rms}


def subsequence_dtw(q: np.ndarray, r: np.ndarray, band: int | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    n, m = len(q), len(r)
    cost = (q[:, None] - r[None, :]) ** 2
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    ptr = np.zeros((n + 1, m + 1), dtype=np.uint8)
    dp[0, :] = 0.0  # open begin in reference
    for i in range(1, n + 1):
        if band is None:
            lo, hi = 1, m + 1
        else:
            center = int(round((i - 1) * (m - 1) / max(n - 1, 1))) + 1
            lo, hi = max(1, center - band), min(m + 1, center + band + 1)
        for j in range(lo, hi):
            choices = (dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            k = int(np.argmin(choices))
            dp[i, j] = cost[i - 1, j - 1] + choices[k]
            ptr[i, j] = k
    j = int(np.argmin(dp[n, 1:]) + 1)
    best = float(dp[n, j] / max(n, 1))
    i = n
    qi = []
    rj = []
    while i > 0 and j > 0:
        qi.append(i - 1)
        rj.append(j - 1)
        move = ptr[i, j]
        if move == 0:
            i -= 1
        elif move == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    qi = np.asarray(qi[::-1], dtype=int)
    rj = np.asarray(rj[::-1], dtype=int)
    # Keep one reference assignment per query index by taking the median of repeats.
    mapped = []
    for idx in range(n):
        refs = rj[qi == idx]
        if len(refs):
            mapped.append(int(np.median(refs)))
        else:
            mapped.append(mapped[-1] if mapped else 0)
    return np.arange(n), np.asarray(mapped, dtype=int), best


def distance_banded_dtw(
    q_s: np.ndarray,
    q: np.ndarray,
    r_s: np.ndarray,
    r: np.ndarray,
    band_m: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    n, m = len(q), len(r)
    cost = (q[:, None] - r[None, :]) ** 2
    allowed = np.abs(q_s[:, None] - r_s[None, :]) <= band_m
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    ptr = np.zeros((n + 1, m + 1), dtype=np.uint8)
    first_allowed = np.flatnonzero(allowed[0])
    if len(first_allowed) == 0:
        return np.arange(n), np.full(n, -1, dtype=int), math.inf
    dp[0, first_allowed + 1] = 0.0
    for i in range(1, n + 1):
        js = np.flatnonzero(allowed[i - 1]) + 1
        for j in js:
            choices = (dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            k = int(np.argmin(choices))
            best = choices[k]
            if not np.isfinite(best):
                continue
            dp[i, j] = cost[i - 1, j - 1] + best
            ptr[i, j] = k
    last_allowed = np.flatnonzero(allowed[-1]) + 1
    if len(last_allowed) == 0 or not np.isfinite(dp[n, last_allowed]).any():
        return np.arange(n), np.full(n, -1, dtype=int), math.inf
    j = int(last_allowed[np.argmin(dp[n, last_allowed])])
    best = float(dp[n, j] / max(n, 1))
    i = n
    qi = []
    rj = []
    while i > 0 and j > 0:
        if not allowed[i - 1, j - 1]:
            break
        qi.append(i - 1)
        rj.append(j - 1)
        move = ptr[i, j]
        if move == 0:
            i -= 1
        elif move == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    qi = np.asarray(qi[::-1], dtype=int)
    rj = np.asarray(rj[::-1], dtype=int)
    mapped = []
    for idx in range(n):
        refs = rj[qi == idx]
        if len(refs):
            mapped.append(int(np.median(refs)))
        else:
            nearest = int(np.argmin(np.abs(r_s - q_s[idx])))
            mapped.append(nearest)
    return np.arange(n), np.asarray(mapped, dtype=int), best


def fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    return float(a), float(b), pred


def run() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref_s, ref_features, _ = ac.build_reference_features("fwd_z_y_x_back_z_y_minusx", "all")
    ref_z = z_valid(ref_features["total_raw_hp"])
    queries = ac.build_query_features("fwd_z_y_x_back_z_y_minusx")
    rows = []
    for q in queries:
        arr = q.features["total_raw_hp"]
        mask = np.isfinite(arr)
        if mask.sum() < 100:
            continue
        q_s = q.distance[mask]
        q_z = z_valid(arr[mask])
        # Downsample for stable DTW and keep monotonic distance order.
        step = max(1, int(round(len(q_s) / 500)))
        q_s_ds = q_s[::step]
        q_z_ds = q_z[::step]
        ref_step = max(1, int(round(len(ref_s) / 900)))
        ref_s_ds = ref_s[::ref_step]
        ref_z_ds = ref_z[::ref_step]

        ident = identity_metrics(q_s_ds, q_z_ds, ref_s_ds, ref_z_ds)
        qi, rj, dtw_cost = subsequence_dtw(q_z_ds, ref_z_ds)
        mapped_s = ref_s_ds[rj]
        aligned_z = ref_z_ds[rj]
        _, brj, bdtw_cost = distance_banded_dtw(q_s_ds, q_z_ds, ref_s_ds, ref_z_ds, band_m=60.0)
        valid_b = brj >= 0
        bmapped_s = ref_s_ds[np.clip(brj, 0, len(ref_s_ds) - 1)]
        baligned_z = ref_z_ds[np.clip(brj, 0, len(ref_s_ds) - 1)]
        corr = float(np.corrcoef(q_z_ds, aligned_z)[0, 1]) if len(q_z_ds) > 2 else math.nan
        rms = float(np.sqrt(np.mean((q_z_ds - aligned_z) ** 2)))
        bcorr = float(np.corrcoef(q_z_ds[valid_b], baligned_z[valid_b])[0, 1]) if valid_b.sum() > 2 else math.nan
        brms = float(np.sqrt(np.mean((q_z_ds[valid_b] - baligned_z[valid_b]) ** 2))) if valid_b.any() else math.nan
        abs_axis_err = np.abs(mapped_s - q_s_ds)
        b_abs_axis_err = np.abs(bmapped_s[valid_b] - q_s_ds[valid_b]) if valid_b.any() else np.asarray([np.nan])
        scale, offset, affine_s = fit_affine(q_s_ds, mapped_s)
        affine_res = mapped_s - affine_s
        row = {
            "segment_label": q.segment,
            "direction": q.direction,
            "point_count": int(len(q_s_ds)),
            "distance_start_m": float(np.nanmin(q_s_ds)),
            "distance_end_m": float(np.nanmax(q_s_ds)),
            **ident,
            "dtw_corr": corr,
            "dtw_rms_z": rms,
            "dtw_cost": dtw_cost,
            "band60_dtw_corr": bcorr,
            "band60_dtw_rms_z": brms,
            "band60_dtw_cost": bdtw_cost,
            "band60_axis_median_abs_diff_m": float(np.nanmedian(b_abs_axis_err)),
            "band60_axis_p90_abs_diff_m": float(np.nanpercentile(b_abs_axis_err, 90)),
            "dtw_axis_median_abs_diff_m": float(np.median(abs_axis_err)),
            "dtw_axis_p90_abs_diff_m": float(np.percentile(abs_axis_err, 90)),
            "affine_scale_ref_per_query": scale,
            "affine_offset_m": offset,
            "affine_residual_median_abs_m": float(np.median(np.abs(affine_res))),
            "affine_residual_p90_abs_m": float(np.percentile(np.abs(affine_res), 90)),
        }
        rows.append(row)
        plot_segment(q.segment, q_s_ds, q_z_ds, ref_s_ds, ref_z_ds, mapped_s, aligned_z)
    summary = pd.DataFrame(rows).sort_values("dtw_rms_z")
    summary.to_csv(OUT_DIR / "distance_warp_diagnostic_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "distance_warp_diagnostic_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_notes(summary, OUT_DIR / "distance_warp_diagnostic_notes.md")
    print(summary.round(3).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


def plot_segment(segment: str, q_s: np.ndarray, q_z: np.ndarray, ref_s: np.ndarray, ref_z: np.ndarray, mapped_s: np.ndarray, aligned_z: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=180, sharex=False)
    axes[0].plot(ref_s, ref_z, color="#777777", lw=1.2, label="4.14 reference total HP")
    axes[0].plot(q_s, q_z, color="#d62728", lw=1.2, label="5.13 query on SPAN distance")
    axes[0].set_title(f"{segment}: identity distance comparison")
    axes[0].set_xlabel("SPAN/projected distance / m")
    axes[0].set_ylabel("robust z")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].plot(q_s, q_z, color="#d62728", lw=1.1, label="query")
    axes[1].plot(q_s, aligned_z, color="#1f77b4", lw=1.1, label="reference after DTW mapped to query index")
    ax2 = axes[1].twinx()
    ax2.plot(q_s, mapped_s - q_s, color="#2ca02c", alpha=0.55, lw=1.0, label="mapped_s - SPAN_s")
    axes[1].set_title("DTW-aligned magnetic shape and implied distance-axis correction")
    axes[1].set_xlabel("query SPAN/projected distance / m")
    axes[1].set_ylabel("robust z")
    ax2.set_ylabel("distance correction / m")
    axes[1].grid(alpha=0.25)
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"distance_warp_{segment}.png")
    plt.close(fig)


def write_notes(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Distance-axis Warp Diagnostic",
        "",
        "Purpose: test whether cross-day failures come from magnetic non-repeatability or from SPAN/projected distance-axis errors.",
        "",
        "This is not a deployable localization result. DTW uses the whole segment and is allowed to warp the distance axis monotonically.",
        "",
        "Summary:",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Interpretation:",
        "",
        "- If DTW correlation is high while identity correlation is low, the magnetic shape is repeatable but the distance axis is shifted or warped.",
        "- If the implied distance correction is large or non-affine, the current SPAN/GPGGA projection may not be a clean ground truth for meter-level cross-day evaluation.",
        "- If DTW is still poor, the magnetic signature itself is not stable enough and method work should focus on robust features / map quality.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
