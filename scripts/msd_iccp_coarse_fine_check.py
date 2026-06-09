from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PROC = Path(r"C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\data_proc_new")
OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\msd_iccp_check")


CHANNELS_BODY = {
    "x": "mag_x",
    "y": "mag_y",
    "z": "mag_z",
    "total": "mag_total",
}

CHANNELS_TRACK_ANOM = {
    "x_track_anom": "mag_x_track_anom",
    "y_track_anom": "mag_y_track_anom",
    "z_track_anom": "mag_z_track_anom",
    "total": "mag_total",
}


@dataclass(frozen=True)
class MatchResult:
    case: str
    ref_date: str
    query_date: str
    ref_segment: str
    query_segment: str
    feature_set: str
    window_start_m: float
    window_end_m: float
    coarse_offset_m: float
    fine_offset_m: float
    true_offset_m: float
    coarse_abs_error_m: float
    fine_abs_error_m: float
    fine_score_msd: float
    corr_mean: float
    rms_raw_mean_nT: float
    rms_bias_removed_mean_nT: float
    valid_points: int


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        scale = np.nanstd(x)
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return (x - med) / scale


def moving_average(x: np.ndarray, n: int = 5) -> np.ndarray:
    if n <= 1:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(n, center=True, min_periods=1).mean().to_numpy()


def load_map(date_tag: str) -> pd.DataFrame:
    path = DATA_PROC / f"magmap_{date_tag}_0p5m.csv"
    df = pd.read_csv(path)
    df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")
    return df


def load_segments(date_tag: str) -> pd.DataFrame:
    path = DATA_PROC / f"magmap_{date_tag}_segments.csv"
    return pd.read_csv(path)


def segment_series(
    df: pd.DataFrame,
    segment: str,
    channels: dict[str, str],
    start_m: float,
    end_m: float,
    step_m: float = 1.0,
    smooth_points: int = 3,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    grid = np.arange(start_m, end_m + 1e-9, step_m)
    out: dict[str, np.ndarray] = {}
    dist = df["distance_m"].to_numpy(dtype=float)
    for name, suffix in channels.items():
        col = f"{segment}_{suffix}"
        if col not in df.columns:
            out[name] = np.full_like(grid, np.nan, dtype=float)
            continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(dist) & np.isfinite(vals)
        if mask.sum() < 2:
            out[name] = np.full_like(grid, np.nan, dtype=float)
            continue
        interp = np.interp(grid, dist[mask], vals[mask], left=np.nan, right=np.nan)
        valid_orig = (grid >= dist[mask].min()) & (grid <= dist[mask].max())
        interp[~valid_orig] = np.nan
        out[name] = moving_average(interp, smooth_points)
    return grid, out


def fused_series(
    df: pd.DataFrame,
    channels: dict[str, str],
    start_m: float,
    end_m: float,
    step_m: float = 1.0,
    smooth_points: int = 3,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    grid = np.arange(start_m, end_m + 1e-9, step_m)
    out: dict[str, np.ndarray] = {}
    dist = df["distance_m"].to_numpy(dtype=float)
    for name, suffix in channels.items():
        if suffix == "mag_total":
            col = "map_mag_total_mean_nT"
        elif suffix.startswith("mag_") and suffix.endswith("_track_anom"):
            col = f"map_{suffix}_mean_nT"
        else:
            col = f"map_{suffix}_mean_nT"
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(dist) & np.isfinite(vals)
        interp = np.interp(grid, dist[mask], vals[mask], left=np.nan, right=np.nan)
        valid_orig = (grid >= dist[mask].min()) & (grid <= dist[mask].max())
        interp[~valid_orig] = np.nan
        out[name] = moving_average(interp, smooth_points)
    return grid, out


def stack_features(series: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.vstack([series[name] for name in names]).T


def score_shift(
    q_local: np.ndarray,
    q_feat_raw: np.ndarray,
    ref_s: np.ndarray,
    ref_feat_raw: np.ndarray,
    offset: float,
    min_points: int = 30,
) -> tuple[float, int]:
    ref_at_q = np.column_stack(
        [
            np.interp(q_local + offset, ref_s, ref_feat_raw[:, j], left=np.nan, right=np.nan)
            for j in range(ref_feat_raw.shape[1])
        ]
    )
    mask = np.isfinite(q_feat_raw).all(axis=1) & np.isfinite(ref_at_q).all(axis=1)
    if mask.sum() < min_points:
        return math.inf, int(mask.sum())
    errs = []
    for j in range(q_feat_raw.shape[1]):
        qz = robust_z(q_feat_raw[mask, j])
        rz = robust_z(ref_at_q[mask, j])
        errs.append((qz - rz) ** 2)
    return float(np.mean(np.column_stack(errs))), int(mask.sum())


def coarse_msd(
    q_local: np.ndarray,
    q_feat: np.ndarray,
    ref_s: np.ndarray,
    ref_feat: np.ndarray,
    search_min: float,
    search_max: float,
    step: float = 1.0,
) -> tuple[float, float, int]:
    best = (math.inf, math.nan, 0)
    for off in np.arange(search_min, search_max + 1e-9, step):
        score, n = score_shift(q_local, q_feat, ref_s, ref_feat, float(off))
        if score < best[0]:
            best = (score, float(off), n)
    return best[1], best[0], best[2]


def iccp_1d_refine(
    q_local: np.ndarray,
    q_feat_raw: np.ndarray,
    ref_s: np.ndarray,
    ref_feat_raw: np.ndarray,
    init_offset: float,
    gate_m: float = 8.0,
    max_iter: int = 10,
) -> float:
    # This is a rail-specific 1D ICCP-style refinement. The state is only along-track
    # translation, because rotation/2D heading is not observable after projection to a
    # known rail centerline.
    offset = float(init_offset)
    q_valid = np.isfinite(q_feat_raw).all(axis=1)
    ref_valid = np.isfinite(ref_feat_raw).all(axis=1)
    ref_s_valid = ref_s[ref_valid]
    ref_feat_valid = ref_feat_raw[ref_valid]
    if q_valid.sum() < 30 or ref_valid.sum() < 30:
        return offset

    for _ in range(max_iter):
        deltas: list[float] = []
        q_idx = np.where(q_valid)[0]
        for i in q_idx:
            pred_s = q_local[i] + offset
            local = np.abs(ref_s_valid - pred_s) <= gate_m
            if local.sum() < 2:
                continue
            cand_feat = ref_feat_valid[local]
            qv = q_feat_raw[i]
            # Per-sample feature distance. Use local robust feature scaling so one
            # large-axis component cannot dominate all correspondences.
            scale = np.nanstd(cand_feat, axis=0)
            scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
            d2 = np.sum(((cand_feat - qv) / scale) ** 2, axis=1)
            best_local = np.argmin(d2)
            matched_s = ref_s_valid[local][best_local]
            deltas.append(float(matched_s - q_local[i]))
        if len(deltas) < 10:
            break
        d = np.asarray(deltas)
        med = np.median(d)
        spread = 1.4826 * np.median(np.abs(d - med))
        if not np.isfinite(spread) or spread < 1e-6:
            spread = np.std(d) if np.std(d) > 1e-6 else 1.0
        keep = np.abs(d - med) <= 2.5 * spread
        new_offset = float(np.mean(d[keep])) if keep.any() else float(med)
        if abs(new_offset - offset) < 0.05:
            offset = new_offset
            break
        offset = new_offset

    # Sub-meter local MSD polish around the ICCP-style correspondence estimate.
    best_off, _, _ = coarse_msd(
        q_local,
        q_feat_raw,
        ref_s,
        ref_feat_raw,
        offset - 2.0,
        offset + 2.0,
        step=0.1,
    )
    return float(best_off)


def metrics_at_shift(
    q_local: np.ndarray,
    q_series: dict[str, np.ndarray],
    ref_s: np.ndarray,
    ref_series: dict[str, np.ndarray],
    channels: list[str],
    offset: float,
) -> tuple[dict[str, dict[str, float]], int]:
    metrics: dict[str, dict[str, float]] = {}
    common_count = 0
    for ch in channels:
        q = q_series[ch]
        r = np.interp(q_local + offset, ref_s, ref_series[ch], left=np.nan, right=np.nan)
        mask = np.isfinite(q) & np.isfinite(r)
        common_count = max(common_count, int(mask.sum()))
        if mask.sum() < 3:
            metrics[ch] = {
                "corr": np.nan,
                "rms_raw_nT": np.nan,
                "rms_bias_removed_nT": np.nan,
                "bias_nT": np.nan,
            }
            continue
        diff = q[mask] - r[mask]
        bias = float(np.mean(diff))
        corr = float(np.corrcoef(q[mask], r[mask])[0, 1])
        metrics[ch] = {
            "corr": corr,
            "rms_raw_nT": float(np.sqrt(np.mean(diff**2))),
            "rms_bias_removed_nT": float(np.sqrt(np.mean((diff - bias) ** 2))),
            "bias_nT": bias,
        }
    return metrics, common_count


def plot_case(
    out_path: Path,
    title: str,
    q_local: np.ndarray,
    q_series: dict[str, np.ndarray],
    ref_s: np.ndarray,
    ref_series: dict[str, np.ndarray],
    channels: list[str],
    offset: float,
    metrics: dict[str, dict[str, float]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    axes = axes.ravel()
    labels = {"x": "X", "y": "Y", "z": "Z", "total": "TOTAL", "x_track_anom": "X track anomaly", "y_track_anom": "Y track anomaly", "z_track_anom": "Z track anomaly"}
    for ax, ch in zip(axes, channels):
        x_ref = q_local + offset
        r = np.interp(x_ref, ref_s, ref_series[ch], left=np.nan, right=np.nan)
        q = q_series[ch]
        mask = np.isfinite(q) & np.isfinite(r)
        ax.plot(x_ref[mask], r[mask], lw=1.3, label="reference")
        ax.plot(x_ref[mask], q[mask], lw=1.2, label="query matched")
        m = metrics[ch]
        ax.set_title(
            f"{labels.get(ch, ch)} | RMS={m['rms_raw_nT']:.1f} nT, "
            f"RMS-bias={m['rms_bias_removed_nT']:.1f} nT, Corr={m['corr']:.3f}"
        )
        ax.set_xlabel("Distance along reference track / m")
        ax.set_ylabel("Magnetic field / nT")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_case(
    case: str,
    ref_date: str,
    query_date: str,
    ref_segment: str,
    query_segment: str,
    feature_set: str,
    channels: dict[str, str],
    window: tuple[float, float],
    search_range: tuple[float, float] = (0.0, 560.0),
    step_m: float = 1.0,
) -> MatchResult:
    ref_df = load_map(ref_date)
    query_df = load_map(query_date)
    names = list(channels.keys())
    win_start, win_end = window

    ref_s, ref_series = segment_series(ref_df, ref_segment, channels, 0.0, 560.0, step_m=step_m)
    q_abs, q_series_abs = segment_series(query_df, query_segment, channels, win_start, win_end, step_m=step_m)
    q_local = q_abs - q_abs[0]
    q_feat = stack_features(q_series_abs, names)
    ref_feat = stack_features(ref_series, names)

    coarse_offset, _, _ = coarse_msd(
        q_local,
        q_feat,
        ref_s,
        ref_feat,
        search_range[0],
        search_range[1] - (win_end - win_start),
        step=1.0,
    )
    fine_offset = iccp_1d_refine(q_local, q_feat, ref_s, ref_feat, coarse_offset)
    fine_score, _ = score_shift(q_local, q_feat, ref_s, ref_feat, fine_offset)
    metrics, valid_points = metrics_at_shift(q_local, q_series_abs, ref_s, ref_series, names, fine_offset)
    plot_case(
        OUT_DIR / f"{case}_{feature_set}_{ref_segment}_vs_{query_segment}.png",
        f"{case}: {ref_date}/{ref_segment} vs {query_date}/{query_segment}; offset={fine_offset:.2f} m",
        q_local,
        q_series_abs,
        ref_s,
        ref_series,
        names[:4],
        fine_offset,
        metrics,
    )
    corr_mean = float(np.nanmean([metrics[ch]["corr"] for ch in names]))
    rms_raw_mean = float(np.nanmean([metrics[ch]["rms_raw_nT"] for ch in names]))
    rms_bias_mean = float(np.nanmean([metrics[ch]["rms_bias_removed_nT"] for ch in names]))
    true_offset = win_start
    return MatchResult(
        case=case,
        ref_date=ref_date,
        query_date=query_date,
        ref_segment=ref_segment,
        query_segment=query_segment,
        feature_set=feature_set,
        window_start_m=win_start,
        window_end_m=win_end,
        coarse_offset_m=float(coarse_offset),
        fine_offset_m=float(fine_offset),
        true_offset_m=true_offset,
        coarse_abs_error_m=abs(float(coarse_offset) - true_offset),
        fine_abs_error_m=abs(float(fine_offset) - true_offset),
        fine_score_msd=float(fine_score),
        corr_mean=corr_mean,
        rms_raw_mean_nT=rms_raw_mean,
        rms_bias_removed_mean_nT=rms_bias_mean,
        valid_points=valid_points,
    )


def segment_coverage_summary() -> pd.DataFrame:
    rows = []
    for date in ("4_14", "5_13"):
        segs = load_segments(date)
        for _, row in segs.iterrows():
            rows.append(
                {
                    "date": date,
                    "segment": row["segment_label"],
                    "direction": row["direction"],
                    "s_min_m": row["s_min_m"],
                    "s_max_m": row["s_max_m"],
                    "samples": row["samples"],
                    "mag_total_mean_nT": row["mag_total_mean_nT"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coverage = segment_coverage_summary()
    coverage.to_csv(OUT_DIR / "segment_coverage_summary.csv", index=False, encoding="utf-8-sig")

    cases = [
        # Screenshot-like check: two single 4.14 forward passes, raw body axes,
        # around the same 75-265 m window shown in the image.
        (
            "same_day_4_14_forward_body_75_265",
            "4_14",
            "4_14",
            "BMAW15230010L_3_seg03",
            "BMAW15230010L_5_seg01",
            "body_raw",
            CHANNELS_BODY,
            (75.0, 265.0),
        ),
        (
            "same_day_4_14_backward_body_75_238",
            "4_14",
            "4_14",
            "BMAW15230010L_3_seg02",
            "BMAW15230010L_2_seg01",
            "body_raw",
            CHANNELS_BODY,
            (75.0, 238.0),
        ),
        (
            "cross_day_forward_body_75_265",
            "4_14",
            "5_13",
            "BMAW15230010L_3_seg03",
            "BMAW15230010L_1_seg04",
            "body_raw",
            CHANNELS_BODY,
            (75.0, 265.0),
        ),
        (
            "cross_day_forward_body_90_265",
            "4_14",
            "5_13",
            "BMAW15230010L_3_seg03",
            "BMAW15230010L_9_seg02",
            "body_raw",
            CHANNELS_BODY,
            (90.0, 265.0),
        ),
        (
            "cross_day_backward_body_75_238",
            "4_14",
            "5_13",
            "BMAW15230010L_3_seg02",
            "BMAW15230010L_1_seg03",
            "body_raw",
            CHANNELS_BODY,
            (75.0, 238.0),
        ),
        # Direction-corrected/anomaly features used by our pipeline.
        (
            "cross_day_forward_trackanom_75_265",
            "4_14",
            "5_13",
            "BMAW15230010L_3_seg03",
            "BMAW15230010L_1_seg04",
            "track_anom",
            CHANNELS_TRACK_ANOM,
            (75.0, 265.0),
        ),
    ]

    results = []
    for args in cases:
        try:
            results.append(run_case(*args))
        except Exception as exc:
            print(f"case failed: {args[0]}: {exc}")

    result_rows = [r.__dict__ for r in results]
    pd.DataFrame(result_rows).to_csv(OUT_DIR / "msd_iccp_case_summary.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "msd_iccp_case_summary.json", "w", encoding="utf-8") as f:
        json.dump(result_rows, f, ensure_ascii=False, indent=2)
    print(pd.DataFrame(result_rows).to_string(index=False))
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
