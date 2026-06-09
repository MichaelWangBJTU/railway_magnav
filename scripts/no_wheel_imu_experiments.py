from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path.home() / "Desktop" / "磁导航" / "数据" / "codex_railway_magnav"
PROC_DIR = PROJECT_ROOT / "data_proc_new"
DATA_ROOT = PROJECT_ROOT / "data"
OUT_ROOT = PROJECT_ROOT / "no_wheel_imu"
PROJECT_ROOT = Path.home() / "Desktop" / "\u78c1\u5bfc\u822a" / "\u6570\u636e" / "codex_railway_magnav"
PROC_DIR = PROJECT_ROOT / "data_proc_new"
DATA_ROOT = PROJECT_ROOT / "data"
OUT_ROOT = PROJECT_ROOT / "no_wheel_imu"
STEP_M = 0.5
GPS_UTC_LEAP_SECONDS = 18
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "code": root / "code",
        "outputs": root / "outputs",
        "figures": root / "figures",
        "reports": root / "reports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def gps_week_seconds_to_beijing(week: int, sow: float) -> pd.Timestamp:
    utc = GPS_EPOCH + timedelta(weeks=int(week), seconds=float(sow) - GPS_UTC_LEAP_SECONDS)
    local = utc + timedelta(hours=8)
    return pd.Timestamp(local.replace(tzinfo=None))


def clean_crc(text: str) -> str:
    return text.split("*", 1)[0]


def parse_inspvax_file(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#INSPVAX"):
                continue
            if ";" not in line:
                continue
            head, body = line.split(";", 1)
            hp = head.split(",")
            bp = body.split(",")
            if len(hp) < 7 or len(bp) < 22:
                continue
            try:
                week = int(hp[5])
                sow = float(hp[6])
                rows.append(
                    {
                        "time": gps_week_seconds_to_beijing(week, sow),
                        "week": week,
                        "sow": sow,
                        "ins_status": bp[0],
                        "pos_type": bp[1],
                        "lat": float(bp[2]),
                        "lon": float(bp[3]),
                        "height_m": float(bp[4]),
                        "north_vel_mps": float(bp[6]),
                        "east_vel_mps": float(bp[7]),
                        "up_vel_mps": float(bp[8]),
                        "roll_deg": float(bp[9]),
                        "pitch_deg": float(bp[10]),
                        "azimuth_deg": float(bp[11]),
                        "north_vel_std_mps": float(bp[15]),
                        "east_vel_std_mps": float(bp[16]),
                        "up_vel_std_mps": float(bp[17]),
                        "roll_std_deg": float(bp[18]),
                        "pitch_std_deg": float(bp[19]),
                        "azimuth_std_deg": float(bp[20]),
                        "source_file": str(path),
                        "span_label": path.name.replace("_INSPVAX.ASCII", ""),
                    }
                )
            except (ValueError, IndexError):
                continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["horiz_speed_mps"] = np.hypot(df["north_vel_mps"], df["east_vel_mps"])
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df


def parse_all_inspvax(data_root: Path, out_dirs: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for path in data_root.rglob("*_INSPVAX.ASCII"):
        if "Converted_On_20260608" not in str(path):
            continue
        df = parse_inspvax_file(path)
        if not df.empty:
            frames.append(df)
    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not all_df.empty:
        all_df = all_df.sort_values("time").reset_index(drop=True)
        all_df.to_csv(out_dirs["outputs"] / "inspvax_parsed_summary.csv", index=False, encoding="utf-8-sig")
    return all_df


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-9:
        return np.full_like(x, np.nan)
    return (x - mu) / sd


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-9:
        return zscore(x)
    return (x - med) / (1.4826 * mad)


def corrcoef(a: np.ndarray, b: np.ndarray, min_valid_ratio: float = 0.8) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        return math.nan
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < max(12, int(len(a) * min_valid_ratio)):
        return math.nan
    az = robust_zscore(a[mask])
    bz = robust_zscore(b[mask])
    if not np.isfinite(az).all() or not np.isfinite(bz).all():
        return math.nan
    az = np.clip(az, -6.0, 6.0)
    bz = np.clip(bz, -6.0, 6.0)
    az = az - np.nanmean(az)
    bz = bz - np.nanmean(bz)
    denom = math.sqrt(float(np.nanmean(az * az) * np.nanmean(bz * bz)))
    if not np.isfinite(denom) or denom < 1e-9:
        return math.nan
    return float(np.nanmean(az * bz) / denom)


def rolling_nanmedian(x: np.ndarray, points: int) -> np.ndarray:
    points = max(3, int(points) | 1)
    return pd.Series(np.asarray(x, dtype=float)).rolling(points, center=True, min_periods=max(3, points // 3)).median().to_numpy(float)


def highpass_by_distance(values: np.ndarray, rel_m: np.ndarray, window_m: float = 30.0) -> np.ndarray:
    rel_m = np.asarray(rel_m, dtype=float)
    values = np.asarray(values, dtype=float)
    diffs = np.diff(rel_m[np.isfinite(rel_m)])
    diffs = diffs[np.abs(diffs) > 1e-6]
    step = float(np.nanmedian(np.abs(diffs))) if len(diffs) else STEP_M
    points = max(5, int(round(window_m / max(step, 0.05))) | 1)
    base = rolling_nanmedian(values, points)
    return values - base


def gradient_by_distance(values: np.ndarray, rel_m: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    rel_m = np.asarray(rel_m, dtype=float)
    if len(values) < 3 or np.nanmax(rel_m) - np.nanmin(rel_m) < 1.0:
        return np.full_like(values, np.nan)
    return np.gradient(values, rel_m)


def finite_interp(x: np.ndarray, y: np.ndarray, xp: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xp = np.asarray(xp, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.full_like(xp, np.nan, dtype=float)
    return np.interp(xp, x[mask], y[mask], left=np.nan, right=np.nan)


def build_reference(proc_dir: Path) -> dict[str, np.ndarray]:
    ref = pd.read_csv(proc_dir / "magmap_4_14_fused_0p5m.csv")
    dist = ref["distance_m"].to_numpy(float)
    total = rolling_nanmedian(ref["mag_total_smooth_nT"].to_numpy(float), int(round(5.0 / STEP_M)))
    hp_total = highpass_by_distance(total, dist, 30.0)
    grad_total = gradient_by_distance(total, dist)
    y = rolling_nanmedian(ref["mag_y_track_anom_smooth_nT"].to_numpy(float), int(round(5.0 / STEP_M)))
    hp_y = highpass_by_distance(y, dist, 30.0)
    return {"distance_m": dist, "total": total, "hp_total": hp_total, "grad_total": grad_total, "hp_y": hp_y}


@dataclass
class SegmentQuery:
    segment_label: str
    direction: str
    time: np.ndarray
    distance_true_m: np.ndarray
    total: np.ndarray
    y: np.ndarray
    true_start_m: float
    true_end_m: float


def read_segments(proc_dir: Path, label: str = "5.13") -> pd.DataFrame:
    df = pd.read_csv(proc_dir / f"magmap_{label.replace('.', '_')}_segments.csv")
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    return df.sort_values("start_time").reset_index(drop=True)


def build_segment_queries(proc_dir: Path, label: str = "5.13") -> list[SegmentQuery]:
    aligned_path = proc_dir / f"magmap_{label.replace('.', '_')}_aligned_samples.csv"
    usecols = ["time", "mag_total", "mag_y_track_anom", "s_abs_m", "segment_label", "direction"]
    aligned = pd.read_csv(aligned_path, usecols=usecols)
    aligned["time"] = pd.to_datetime(aligned["time"], errors="coerce")
    aligned = aligned.dropna(subset=["time", "mag_total", "s_abs_m", "segment_label"])
    aligned = aligned[np.isfinite(aligned["mag_total"]) & np.isfinite(aligned["s_abs_m"])].copy()
    segs = read_segments(proc_dir, label)
    queries = []
    for _, seg in segs.iterrows():
        seg_label = str(seg["segment_label"])
        part = aligned[aligned["segment_label"] == seg_label].copy()
        if part.empty:
            continue
        direction = str(seg["direction"])
        part = part.sort_values("time").set_index("time")
        numeric = part[["mag_total", "mag_y_track_anom", "s_abs_m"]].resample("200ms").median()
        numeric = numeric.interpolate(limit=3, limit_direction="both").dropna(subset=["mag_total", "s_abs_m"])
        if numeric.empty:
            continue
        frame = numeric.reset_index().rename(
            columns={"mag_total": "total", "mag_y_track_anom": "y", "s_abs_m": "distance_m"}
        )
        frame = frame.drop_duplicates("time").reset_index(drop=True)
        if len(frame) < 120:
            continue
        # This no-wheel experiment must use the sensor's time order. SPAN
        # distance is kept only as ground truth for evaluation.
        max_points = 500
        if len(frame) > max_points:
            take = np.linspace(0, len(frame) - 1, max_points).round().astype(int)
            frame = frame.iloc[take].reset_index(drop=True)
        smooth_points = max(3, min(31, int(round(len(frame) / 250)) | 1))
        frame["total"] = rolling_nanmedian(frame["total"].to_numpy(float), smooth_points)
        frame["y"] = rolling_nanmedian(frame["y"].to_numpy(float), smooth_points)
        duration_s = (frame["time"].iloc[-1] - frame["time"].iloc[0]).total_seconds()
        net_length_m = abs(float(frame["distance_m"].iloc[-1]) - float(frame["distance_m"].iloc[0]))
        if duration_s < 60.0 or net_length_m < 60.0:
            continue
        queries.append(
            SegmentQuery(
                segment_label=seg_label,
                direction=direction,
                time=frame["time"].to_numpy(dtype="datetime64[ns]"),
                distance_true_m=frame["distance_m"].to_numpy(float),
                total=frame["total"].to_numpy(float),
                y=frame["y"].to_numpy(float),
                true_start_m=float(frame["distance_m"].iloc[0]),
                true_end_m=float(frame["distance_m"].iloc[-1]),
            )
        )
    return queries


def integrate_speed_for_query(query: SegmentQuery, insp: pd.DataFrame) -> tuple[np.ndarray, dict]:
    t = pd.to_datetime(query.time)
    ts = t.astype("int64") / 1e9
    if insp.empty:
        return np.full(len(t), np.nan), {"speed_coverage": 0.0}
    insp_time = pd.to_datetime(insp["time"]).astype("int64") / 1e9
    speed = insp["horiz_speed_mps"].to_numpy(float)
    az = insp["azimuth_deg"].to_numpy(float)
    vstd = np.hypot(insp["north_vel_std_mps"].to_numpy(float), insp["east_vel_std_mps"].to_numpy(float))
    valid = np.isfinite(insp_time) & np.isfinite(speed)
    if valid.sum() < 2:
        return np.full(len(t), np.nan), {"speed_coverage": 0.0}
    speed_i = np.interp(ts, insp_time[valid], speed[valid], left=np.nan, right=np.nan)
    az_i = np.interp(ts, insp_time[valid], az[valid], left=np.nan, right=np.nan)
    vstd_i = np.interp(ts, insp_time[valid], vstd[valid], left=np.nan, right=np.nan)
    coverage = float(np.isfinite(speed_i).mean())
    rel = np.zeros(len(t), dtype=float)
    for i in range(1, len(t)):
        dt = ts[i] - ts[i - 1]
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(speed_i[i]) or not np.isfinite(speed_i[i - 1]):
            rel[i] = np.nan
        else:
            rel[i] = rel[i - 1] + 0.5 * (speed_i[i] + speed_i[i - 1]) * dt
    stats = {
        "speed_coverage": coverage,
        "mean_speed_mps": float(np.nanmean(speed_i)),
        "median_speed_mps": float(np.nanmedian(speed_i)),
        "mean_azimuth_deg": float(np.nanmean(az_i)),
        "median_vel_std_mps": float(np.nanmedian(vstd_i)),
        "rel_length_imu_m": float(np.nanmax(rel) - np.nanmin(rel)) if np.isfinite(rel).any() else math.nan,
    }
    return rel, stats


def make_query_features(total: np.ndarray, rel: np.ndarray, y: np.ndarray | None = None) -> dict[str, np.ndarray]:
    hp = highpass_by_distance(total, rel, 30.0)
    grad = gradient_by_distance(total, rel)
    features = {"hp_total": hp, "grad_total": grad}
    if y is not None and np.isfinite(y).mean() >= 0.75:
        features["hp_y"] = highpass_by_distance(y, rel, 30.0)
    return features


def score_candidate(query_features: dict[str, np.ndarray], ref: dict[str, np.ndarray], positions: np.ndarray) -> float:
    scores = []
    for f in ["hp_total", "grad_total", "hp_y"]:
        if f not in query_features or f not in ref:
            continue
        rv = finite_interp(ref["distance_m"], ref[f], positions)
        qv = query_features[f]
        if np.isfinite(rv).mean() < 0.85 or np.isfinite(qv).mean() < 0.85:
            return math.nan
        scores.append(corrcoef(qv, rv, 0.85))
    scores = [s for s in scores if np.isfinite(s)]
    if not scores:
        return math.nan
    return float(np.mean(scores))


def evaluate_rel_distance_quality(query: SegmentQuery, rel_imu: np.ndarray) -> dict:
    true_rel = np.abs(query.distance_true_m - query.distance_true_m[0])
    mask = np.isfinite(rel_imu) & np.isfinite(true_rel)
    if mask.sum() < 10:
        return {"rel_rmse_m": math.nan, "rel_scale": math.nan, "true_length_m": float(np.nanmax(true_rel)), "imu_length_m": math.nan}
    a = np.vstack([rel_imu[mask], np.ones(mask.sum())]).T
    scale, offset = np.linalg.lstsq(a, true_rel[mask], rcond=None)[0]
    pred = rel_imu[mask] * scale + offset
    return {
        "rel_rmse_m": float(np.sqrt(np.mean((pred - true_rel[mask]) ** 2))),
        "rel_scale": float(scale),
        "true_length_m": float(np.nanmax(true_rel)),
        "imu_length_m": float(np.nanmax(rel_imu[mask]) - np.nanmin(rel_imu[mask])),
    }


def search_start_for_rel(
    query: SegmentQuery,
    ref: dict[str, np.ndarray],
    rel: np.ndarray,
    method: str,
    scale_values: np.ndarray | None = None,
    start_step_m: float = 1.0,
) -> list[dict]:
    rows = []
    if scale_values is None:
        scale_values = np.array([1.0])
    ref_min = float(np.nanmin(ref["distance_m"]))
    ref_max = float(np.nanmax(ref["distance_m"]))
    sign = 1.0 if query.direction == "forward" else -1.0
    for scale in scale_values:
        rel_s = np.asarray(rel, dtype=float) * float(scale)
        if np.isfinite(rel_s).mean() < 0.8 or np.nanmax(rel_s) - np.nanmin(rel_s) < 30.0:
            continue
        rel_s = rel_s - np.nanmin(rel_s)
        # Keep only finite rel and total values.
        mask = np.isfinite(rel_s) & np.isfinite(query.total)
        if mask.sum() < 80:
            continue
        rel_use = rel_s[mask]
        total_use = query.total[mask]
        y_use = query.y[mask] if len(query.y) == len(query.total) else np.full(mask.sum(), np.nan)
        # Normalize rel ordering and remove repeated distances.
        order = np.argsort(rel_use)
        rel_use = rel_use[order]
        total_use = total_use[order]
        y_use = y_use[order]
        keep = np.r_[True, np.diff(rel_use) > 0.05]
        rel_use = rel_use[keep]
        total_use = total_use[keep]
        y_use = y_use[keep]
        if len(rel_use) < 80:
            continue
        qfeat = make_query_features(total_use, rel_use, y_use)
        length = float(np.nanmax(rel_use))
        if query.direction == "forward":
            starts = np.arange(ref_min, ref_max - length + 0.001, start_step_m)
        else:
            starts = np.arange(ref_min + length, ref_max + 0.001, start_step_m)
        if len(starts) == 0:
            continue
        scores = []
        for start in starts:
            positions = start + sign * rel_use
            scores.append(score_candidate(qfeat, ref, positions))
        scores = np.asarray(scores, dtype=float)
        if not np.isfinite(scores).any():
            continue
        best_idx = int(np.nanargmax(scores))
        pred_start = float(starts[best_idx])
        best = float(scores[best_idx])
        far = np.abs(starts - pred_start) >= 20.0
        second = float(np.nanmax(scores[far])) if far.any() and np.isfinite(scores[far]).any() else math.nan
        margin = best - second if np.isfinite(second) else math.nan
        rows.append(
            {
                "method": method,
                "segment_label": query.segment_label,
                "direction": query.direction,
                "scale": float(scale),
                "rel_length_m": length,
                "true_start_m": query.true_start_m,
                "true_end_m": query.true_end_m,
                "pred_start_m": pred_start,
                "error_m": pred_start - query.true_start_m,
                "abs_error_m": abs(pred_start - query.true_start_m),
                "best_score": best,
                "second_score": second,
                "score_margin": margin,
                "accepted_margin_0p20": bool(np.isfinite(margin) and margin >= 0.20),
            }
        )
    return rows


def time_scale_search(query: SegmentQuery, ref: dict[str, np.ndarray]) -> list[dict]:
    t = pd.to_datetime(query.time).astype("int64") / 1e9
    duration = t - t[0]
    if duration[-1] <= 10:
        return []
    tau = duration / duration[-1]
    true_len = abs(query.true_end_m - query.true_start_m)
    max_len = min(560.0, max(120.0, true_len * 1.5))
    min_len = max(60.0, min(true_len * 0.4, 120.0))
    scale_lengths = np.arange(min_len, max_len + 0.001, 20.0)
    rows = []
    for length in scale_lengths:
        rel = tau * length
        rows.extend(search_start_for_rel(query, ref, rel, "NoWheel_time_scale_search", np.array([1.0]), 4.0))
        if rows and rows[-1]["method"] == "NoWheel_time_scale_search":
            rows[-1]["assumed_length_m"] = float(length)
    if not rows:
        return []
    # Keep only best speed/length candidate for this segment.
    df = pd.DataFrame(rows).sort_values(["best_score", "score_margin"], ascending=[False, False])
    return [df.iloc[0].to_dict()]


def distinctive_window_search(
    query: SegmentQuery,
    ref: dict[str, np.ndarray],
    rel: np.ndarray,
    method: str,
    scale_values: np.ndarray,
    window_lengths_m: tuple[float, ...] = (80.0, 120.0, 180.0, 240.0),
    stride_m: float = 20.0,
    start_step_m: float = 2.0,
) -> list[dict]:
    rows = []
    ref_min = float(np.nanmin(ref["distance_m"]))
    ref_max = float(np.nanmax(ref["distance_m"]))
    sign = 1.0 if query.direction == "forward" else -1.0
    for scale in scale_values:
        rel_s = np.asarray(rel, dtype=float) * float(scale)
        if np.isfinite(rel_s).mean() < 0.8 or np.nanmax(rel_s) - np.nanmin(rel_s) < min(window_lengths_m):
            continue
        rel_s = rel_s - np.nanmin(rel_s)
        mask = np.isfinite(rel_s) & np.isfinite(query.total) & np.isfinite(query.distance_true_m)
        if mask.sum() < 120:
            continue
        rel_use = rel_s[mask]
        total_use = query.total[mask]
        y_use = query.y[mask] if len(query.y) == len(query.total) else np.full(mask.sum(), np.nan)
        true_use = query.distance_true_m[mask]
        order = np.argsort(rel_use)
        rel_use = rel_use[order]
        total_use = total_use[order]
        y_use = y_use[order]
        true_use = true_use[order]
        keep = np.r_[True, np.diff(rel_use) > 0.05]
        rel_use = rel_use[keep]
        total_use = total_use[keep]
        y_use = y_use[keep]
        true_use = true_use[keep]
        if len(rel_use) < 120:
            continue
        length = float(np.nanmax(rel_use))
        for win_len in window_lengths_m:
            if length < win_len:
                continue
            offsets = np.arange(0.0, length - win_len + 0.001, stride_m)
            for offset in offsets:
                wmask = (rel_use >= offset) & (rel_use <= offset + win_len)
                if wmask.sum() < 80:
                    continue
                rel_w = rel_use[wmask]
                total_w = total_use[wmask]
                y_w = y_use[wmask]
                true_w = true_use[wmask]
                rel_w = rel_w - rel_w[0]
                actual_len = float(np.nanmax(rel_w))
                if actual_len < win_len * 0.75:
                    continue
                qfeat = make_query_features(total_w, rel_w, y_w)
                if query.direction == "forward":
                    starts = np.arange(ref_min, ref_max - actual_len + 0.001, start_step_m)
                else:
                    starts = np.arange(ref_min + actual_len, ref_max + 0.001, start_step_m)
                if len(starts) == 0:
                    continue
                scores = []
                for start in starts:
                    positions = start + sign * rel_w
                    scores.append(score_candidate(qfeat, ref, positions))
                scores = np.asarray(scores, dtype=float)
                if not np.isfinite(scores).any():
                    continue
                best_idx = int(np.nanargmax(scores))
                pred_start = float(starts[best_idx])
                best = float(scores[best_idx])
                far = np.abs(starts - pred_start) >= 20.0
                second = float(np.nanmax(scores[far])) if far.any() and np.isfinite(scores[far]).any() else math.nan
                margin = best - second if np.isfinite(second) else math.nan
                true_start = float(true_w[0])
                rows.append(
                    {
                        "method": method,
                        "segment_label": query.segment_label,
                        "direction": query.direction,
                        "scale": float(scale),
                        "rel_length_m": actual_len,
                        "window_offset_m": float(offset),
                        "window_length_m": float(win_len),
                        "true_start_m": true_start,
                        "true_end_m": float(true_w[-1]),
                        "pred_start_m": pred_start,
                        "error_m": pred_start - true_start,
                        "abs_error_m": abs(pred_start - true_start),
                        "best_score": best,
                        "second_score": second,
                        "score_margin": margin,
                        "accepted_margin_0p20": bool(np.isfinite(margin) and margin >= 0.20),
                    }
                )
    if not rows:
        return []
    df = pd.DataFrame(rows)
    # Prefer distinctive, high-correlation windows; a small length bonus avoids
    # picking very short accidental matches when scores are nearly tied.
    df["selection_score"] = (
        df["score_margin"].fillna(-9.0)
        + 0.25 * df["best_score"].fillna(-9.0)
        + 0.0005 * df["rel_length_m"].fillna(0.0)
    )
    df = df.sort_values(["selection_score", "score_margin", "best_score"], ascending=[False, False, False])
    return [df.iloc[0].drop(labels=["selection_score"]).to_dict()]


def run_experiments(proc_dir: Path, data_root: Path, out_root: Path) -> None:
    setup_matplotlib()
    out_dirs = ensure_dirs(out_root)
    insp = parse_all_inspvax(data_root, out_dirs)
    ref = build_reference(proc_dir)
    queries = build_segment_queries(proc_dir, "5.13")

    rows = []
    rel_quality_rows = []
    for q in queries:
        rel_imu, speed_stats = integrate_speed_for_query(q, insp)
        quality = evaluate_rel_distance_quality(q, rel_imu)
        rel_quality_rows.append({"segment_label": q.segment_label, "direction": q.direction, **speed_stats, **quality})

        rows.extend(time_scale_search(q, ref))
        rows.extend(search_start_for_rel(q, ref, rel_imu, "INSPVAX_speed_integral", np.array([1.0]), 1.0))
        rows.extend(search_start_for_rel(q, ref, rel_imu, "INSPVAX_speed_scaled_search", np.arange(0.3, 2.61, 0.12), 2.0))
        rows.extend(
            distinctive_window_search(
                q,
                ref,
                rel_imu,
                "IMU_distinctive_window",
                np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
                (120.0,),
                80.0,
                8.0,
            )
        )
        rows.extend(
            distinctive_window_search(
                q,
                ref,
                rel_imu,
                "IMU_adaptive_distinctive_window",
                np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.5]),
                (120.0,),
                80.0,
                8.0,
            )
        )

    results = pd.DataFrame(rows)
    if not results.empty:
        keep_best_methods = {
            "INSPVAX_speed_scaled_search",
            "IMU_distinctive_window",
            "IMU_adaptive_distinctive_window",
        }
        best_parts = []
        other = results[~results["method"].isin(keep_best_methods)].copy()
        for method in keep_best_methods:
            part = results[results["method"] == method].copy()
            if part.empty:
                continue
            part = part.sort_values(["segment_label", "best_score", "score_margin"], ascending=[True, False, False])
            best_parts.append(part.groupby("segment_label", as_index=False).head(1))
        results = pd.concat([other, *best_parts], ignore_index=True)
    results.to_csv(out_dirs["outputs"] / "no_wheel_imu_matching_results.csv", index=False, encoding="utf-8-sig")

    rel_quality = pd.DataFrame(rel_quality_rows)
    rel_quality.to_csv(out_dirs["outputs"] / "inspvax_relative_distance_quality.csv", index=False, encoding="utf-8-sig")

    summary = summarize_results(results)
    summary.to_csv(out_dirs["outputs"] / "no_wheel_imu_matching_summary.csv", index=False, encoding="utf-8-sig")
    confidence = summarize_confidence(results)
    confidence.to_csv(out_dirs["outputs"] / "no_wheel_imu_confidence_summary.csv", index=False, encoding="utf-8-sig")
    plot_summary(summary, out_dirs)
    plot_confidence(confidence, out_dirs)
    plot_rel_quality(rel_quality, out_dirs)
    write_json_summary(results, summary, confidence, rel_quality, out_dirs)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    return (
        results.groupby("method")
        .agg(
            segment_count=("abs_error_m", "size"),
            median_abs_error_m=("abs_error_m", "median"),
            mean_abs_error_m=("abs_error_m", "mean"),
            rmse_m=("error_m", lambda x: float(np.sqrt(np.nanmean(np.asarray(x, dtype=float) ** 2)))),
            p90_abs_error_m=("abs_error_m", lambda x: float(np.nanpercentile(x, 90))),
            median_score=("best_score", "median"),
            median_margin=("score_margin", "median"),
            accepted_0p20=("accepted_margin_0p20", "sum"),
        )
        .reset_index()
        .sort_values(["median_abs_error_m", "rmse_m"])
    )


def summarize_confidence(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "score_margin" not in results:
        return pd.DataFrame()
    thresholds = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
    rows = []
    for method, g in results.groupby("method"):
        for th in thresholds:
            sub = g[g["score_margin"] >= th]
            rows.append(
                {
                    "method": method,
                    "margin_threshold": th,
                    "accepted_count": int(len(sub)),
                    "total_count": int(len(g)),
                    "coverage_pct": 100.0 * len(sub) / len(g) if len(g) else 0.0,
                    "median_abs_error_m": float(sub["abs_error_m"].median()) if len(sub) else math.nan,
                    "mean_abs_error_m": float(sub["abs_error_m"].mean()) if len(sub) else math.nan,
                    "p90_abs_error_m": float(np.nanpercentile(sub["abs_error_m"], 90)) if len(sub) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, out_dirs: dict[str, Path]) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=160)
    x = np.arange(len(summary))
    ax.bar(x, summary["median_abs_error_m"], color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"], rotation=18, ha="right")
    ax.set_ylabel("中位绝对误差 / m")
    ax.set_title("无轮速计 + IMU/INS 速度辅助地磁匹配")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dirs["figures"] / "no_wheel_imu_method_summary.png")
    plt.close(fig)


def plot_confidence(conf: pd.DataFrame, out_dirs: dict[str, Path]) -> None:
    if conf.empty:
        return
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=160)
    ax2 = ax1.twinx()
    for method, g in conf.groupby("method"):
        ax1.plot(g["margin_threshold"], g["median_abs_error_m"], marker="o", lw=1.5, label=f"{method} 误差")
        ax2.plot(g["margin_threshold"], g["coverage_pct"], ls="--", lw=1.2, alpha=0.7, label=f"{method} 覆盖率")
    ax1.set_xlabel("唯一性门限 score_best - score_second")
    ax1.set_ylabel("接受片段中位绝对误差 / m")
    ax2.set_ylabel("覆盖率 / %")
    ax1.set_title("无轮速计方案的置信度门控效果")
    ax1.grid(True, alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dirs["figures"] / "no_wheel_imu_confidence_tradeoff.png")
    plt.close(fig)


def plot_rel_quality(relq: pd.DataFrame, out_dirs: dict[str, Path]) -> None:
    if relq.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=160)
    axes[0].bar(relq["segment_label"], relq["true_length_m"], alpha=0.65, label="SPAN 真值长度")
    axes[0].bar(relq["segment_label"], relq["imu_length_m"], alpha=0.65, label="INSPVAX 速度积分长度")
    axes[0].set_ylabel("相对行驶长度 / m")
    axes[0].set_title("INSPVAX 速度积分长度对比")
    axes[0].tick_params(axis="x", labelrotation=45)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(relq["segment_label"], relq["rel_rmse_m"], color="#d62728", alpha=0.8)
    axes[1].set_ylabel("线性校正后相对距离 RMSE / m")
    axes[1].set_title("INSPVAX 相对距离形状误差")
    axes[1].tick_params(axis="x", labelrotation=45)
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dirs["figures"] / "inspvax_relative_distance_quality.png")
    plt.close(fig)


def write_json_summary(results: pd.DataFrame, summary: pd.DataFrame, confidence: pd.DataFrame, relq: pd.DataFrame, out_dirs: dict[str, Path]) -> None:
    payload = {
        "method_summary": summary.to_dict(orient="records") if not summary.empty else [],
        "confidence_summary": confidence.to_dict(orient="records") if not confidence.empty else [],
        "inspvax_relative_distance_quality": relq.to_dict(orient="records") if not relq.empty else [],
        "notes": [
            "No wheel-speed sensor is used. Three variants are compared: magnetic time-scale search without velocity, INSPVAX velocity integration, and INSPVAX velocity integration with scale search.",
            "INSPVAX is an INS/SPAN solution containing north/east/up velocities and attitude. It is not raw accelerometer double integration; in this dataset it may be GNSS-aided.",
            "The localization target is the absolute along-track position at the beginning of each 5.13 segment in time order.",
            "A uniqueness margin score_best - score_second is used to identify ambiguous magnetic matches.",
        ],
    }
    (out_dirs["outputs"] / "no_wheel_imu_experiment_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc-dir", type=Path, default=PROC_DIR)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    run_experiments(args.proc_dir, args.data_root, args.out_root)
    print(json.dumps({"out_root": str(args.out_root), "done": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
