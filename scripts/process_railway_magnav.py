from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(r"C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\data")
OUT_DIR = Path(r"C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\data_proc")

DATASETS = {
    "4.14": {
        "survey_date": datetime(2026, 4, 13),
        "span_dir": BASE / "SPAN4.14",
        "mag_dir": BASE / "mag4.14",
    },
    "5.13": {
        "survey_date": datetime(2026, 5, 13),
        "span_dir": BASE / "SPAN5.13",
        "mag_dir": BASE / "mag5.13",
    },
}

EARTH_R = 6378137.0
GPS_UTC_LEAP_SECONDS = 0


@dataclass
class SpanFile:
    label: str
    path: Path
    df: pd.DataFrame


def gpgga_to_decimal(value: str, hemi: str) -> float:
    if not value:
        return math.nan
    raw = float(value)
    deg = int(raw // 100)
    minutes = raw - deg * 100
    out = deg + minutes / 60.0
    if hemi in ("S", "W"):
        out = -out
    return out


def parse_gpgga_time(local_date: datetime, utc_hhmmss: str, leap_seconds: int) -> datetime:
    raw = float(utc_hhmmss)
    hh = int(raw // 10000)
    mm = int((raw - hh * 10000) // 100)
    ss_float = raw - hh * 10000 - mm * 100
    ss = int(ss_float)
    us = int(round((ss_float - ss) * 1_000_000))
    if us >= 1_000_000:
        ss += 1
        us -= 1_000_000
    utc_dt = local_date.replace(hour=hh, minute=mm, second=ss, microsecond=us)
    if leap_seconds:
        utc_dt = utc_dt - timedelta(seconds=leap_seconds)
    return utc_dt + timedelta(hours=8)


def read_gpgga(path: Path, local_date: datetime, leap_seconds: int) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("$GPGGA"):
                continue
            parts = line.split(",")
            if len(parts) < 10:
                continue
            try:
                t = parse_gpgga_time(local_date, parts[1], leap_seconds)
                lat = gpgga_to_decimal(parts[2], parts[3])
                lon = gpgga_to_decimal(parts[4], parts[5])
                quality = int(parts[6]) if parts[6] else 0
                sats = int(parts[7]) if parts[7] else 0
                hdop = float(parts[8]) if parts[8] else math.nan
                alt = float(parts[9]) if parts[9] else math.nan
            except ValueError:
                continue
            rows.append((t, lat, lon, quality, sats, hdop, alt))
    return pd.DataFrame(
        rows,
        columns=["time", "lat", "lon", "fix_quality", "satellites", "hdop", "alt_m"],
    )


def unique_gpgga_files(span_dir: Path) -> list[Path]:
    candidates = [
        p
        for p in span_dir.rglob("*GPGGA*")
        if p.is_file() and p.stat().st_size > 0 and p.suffix.lower() in {".ascii", ".txt"}
    ]
    # The converter produced root-level copies plus identical files in Converted_On folders.
    # Keep one file per identical name/size pair, preferring the shallower path.
    best: dict[tuple[str, int], Path] = {}
    for p in candidates:
        key = (p.name, p.stat().st_size)
        if key not in best or len(p.parts) < len(best[key].parts):
            best[key] = p
    return sorted(best.values(), key=lambda p: p.name)


def read_mag_file(path: Path) -> pd.DataFrame:
    if path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"Time(yyyymmddHHMMSSzzz.z)": "string"})
    df = df.rename(
        columns={
            "Time(yyyymmddHHMMSSzzz.z)": "time_raw",
            "Mag_X(nT)": "mag_x",
            "Mag_Y(nT)": "mag_y",
            "Mag_Z(nT)": "mag_z",
            "Pitch": "pitch",
            "Roll": "roll",
            "Yaw": "yaw",
        }
    )
    raw = df["time_raw"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(17)
    stamp = raw.str[:14] + raw.str[14:17].str.ljust(6, "0")
    df["time"] = pd.to_datetime(stamp, format="%Y%m%d%H%M%S%f", errors="coerce")
    for col in ["mag_x", "mag_y", "mag_z", "pitch", "roll", "yaw"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["mag_total"] = np.sqrt(df["mag_x"] ** 2 + df["mag_y"] ** 2 + df["mag_z"] ** 2)
    df["mag_file"] = str(path)
    df["mag_label"] = path.stem
    return df.dropna(subset=["time", "mag_x", "mag_y", "mag_z"])


def read_dataset_span(name: str, info: dict, leap_seconds: int) -> list[SpanFile]:
    out = []
    for path in unique_gpgga_files(info["span_dir"]):
        df = read_gpgga(path, info["survey_date"], leap_seconds)
        if df.empty:
            continue
        df = df[df["fix_quality"] > 0].copy()
        df = df.sort_values("time").drop_duplicates("time")
        if len(df) > 1:
            out.append(SpanFile(path.stem.replace("_GPGGA", ""), path, df))
    return out


def read_dataset_mag(info: dict) -> pd.DataFrame:
    frames = []
    for path in sorted(info["mag_dir"].rglob("*.dat")):
        df = read_mag_file(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("time")


def add_local_xy(span_files: list[SpanFile], lat0: float | None = None, lon0: float | None = None) -> tuple[float, float]:
    if lat0 is None or lon0 is None:
        all_lat = np.concatenate([s.df["lat"].to_numpy() for s in span_files])
        all_lon = np.concatenate([s.df["lon"].to_numpy() for s in span_files])
        lat0 = float(np.nanmedian(all_lat))
        lon0 = float(np.nanmedian(all_lon))
    cos0 = math.cos(math.radians(lat0))
    for s in span_files:
        s.df["x_east_m"] = np.radians(s.df["lon"] - lon0) * EARTH_R * cos0
        s.df["y_north_m"] = np.radians(s.df["lat"] - lat0) * EARTH_R
    return lat0, lon0


def fit_track_axis(span_files: list[SpanFile]) -> tuple[np.ndarray, np.ndarray, float]:
    xy = np.vstack(
        [s.df[["x_east_m", "y_north_m"]].to_numpy(dtype=float) for s in span_files]
    )
    center = np.nanmedian(xy, axis=0)
    radius = np.linalg.norm(xy - center, axis=1)
    fit_xy = xy[radius < 1200.0]
    fit_center = np.nanmean(fit_xy, axis=0)
    _, _, vh = np.linalg.svd(fit_xy - fit_center, full_matrices=False)
    axis = vh[0]
    projected = (fit_xy - fit_center) @ axis
    if np.nanpercentile(projected, 95) < abs(np.nanpercentile(projected, 5)):
        axis = -axis
        projected = -projected
    s0 = float(np.nanpercentile(projected, 1))
    s1 = float(np.nanpercentile(projected, 99))
    origin_xy = fit_center + axis * s0
    for s in span_files:
        raw = (s.df[["x_east_m", "y_north_m"]].to_numpy(dtype=float) - origin_xy) @ axis
        s.df["s_abs_m"] = raw
        s.df["span_label"] = s.label
    return axis, origin_xy, s1 - s0


def local_xy_to_latlon(x_east_m: np.ndarray, y_north_m: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    lat = lat0 + np.degrees(y_north_m / EARTH_R)
    lon = lon0 + np.degrees(x_east_m / (EARTH_R * math.cos(math.radians(lat0))))
    return lat, lon


def split_span_by_direction(span_file: SpanFile) -> list[pd.DataFrame]:
    df = span_file.df.copy().sort_values("time").reset_index(drop=True)
    # Smooth position enough to find turn-around points without changing the
    # final 0.5 m map grid. Direction is kept only as a quality label.
    smooth = df["s_abs_m"].rolling(31, center=True, min_periods=1).median().to_numpy()
    ds = np.diff(smooth, prepend=smooth[0])
    direction = np.sign(pd.Series(ds).rolling(15, center=True, min_periods=1).median())
    direction = direction.replace(0, np.nan).ffill().bfill().fillna(0).to_numpy()
    change_idx = np.where(np.diff(direction) != 0)[0] + 1
    cut_points = [0]
    for idx in change_idx:
        if idx - cut_points[-1] >= 50:
            cut_points.append(int(idx))
    cut_points.append(len(df))

    segments = []
    seg_no = 0
    for a, b in zip(cut_points[:-1], cut_points[1:]):
        part = df.iloc[a:b].copy()
        if len(part) < 50:
            continue
        span = float(part["s_abs_m"].max() - part["s_abs_m"].min())
        duration = (part["time"].iloc[-1] - part["time"].iloc[0]).total_seconds()
        if span < 80 or duration < 20:
            continue
        delta_s = float(part["s_abs_m"].iloc[-1] - part["s_abs_m"].iloc[0])
        seg_no += 1
        part["segment_label"] = f"{span_file.label}_seg{seg_no:02d}"
        part["direction"] = "forward" if delta_s >= 0 else "backward"
        segments.append(part)
    return segments


def build_span_segments(span_files: list[SpanFile]) -> list[pd.DataFrame]:
    segments = []
    for sf in span_files:
        segments.extend(split_span_by_direction(sf))
    return segments


def interpolate_segment_to_mag(segment: pd.DataFrame, mag: pd.DataFrame) -> pd.DataFrame:
    start = segment["time"].iloc[0]
    end = segment["time"].iloc[-1]
    # Allow a small edge margin because logger clocks can differ slightly.
    m = mag[(mag["time"] >= start - pd.Timedelta(seconds=1)) & (mag["time"] <= end + pd.Timedelta(seconds=1))].copy()
    if m.empty:
        return pd.DataFrame()

    seg = segment.sort_values("time").drop_duplicates("time")
    t_seg = seg["time"].astype("int64").to_numpy(dtype=float) / 1e9
    t_mag = m["time"].astype("int64").to_numpy(dtype=float) / 1e9
    if len(t_seg) < 2:
        return pd.DataFrame()
    nearest = np.minimum(np.abs(t_mag - t_seg[0]), np.abs(t_mag - t_seg[-1]))
    inside = (t_mag >= t_seg[0]) & (t_mag <= t_seg[-1])
    m = m[inside | (nearest <= 0.2)].copy()
    t_mag = m["time"].astype("int64").to_numpy(dtype=float) / 1e9
    if m.empty:
        return pd.DataFrame()

    for col in ["lat", "lon", "alt_m", "x_east_m", "y_north_m", "s_abs_m", "fix_quality", "satellites", "hdop"]:
        m[col] = np.interp(t_mag, t_seg, seg[col].to_numpy(dtype=float))
    m["span_label"] = seg["span_label"].iloc[0]
    m["segment_label"] = seg["segment_label"].iloc[0]
    m["direction"] = seg["direction"].iloc[0]
    direction_sign = 1.0 if m["direction"].iloc[0] == "forward" else -1.0
    # Body-frame sensor axes: X points to trolley rear, Y down, Z right.
    # Track-frame axes: X along increasing distance, Y down, Z right when facing increasing distance.
    m["mag_x_track"] = -direction_sign * m["mag_x"]
    m["mag_y_track"] = m["mag_y"]
    m["mag_z_track"] = direction_sign * m["mag_z"]
    # For vector components, remove the per-pass body-frame baseline before
    # rotating. This avoids mixing +hard-iron and -hard-iron plateaus when the
    # trolley is turned around, while preserving local magnetic anomaly shapes.
    bx = float(m["mag_x"].median())
    by = float(m["mag_y"].median())
    bz = float(m["mag_z"].median())
    m["mag_x_body_anom"] = m["mag_x"] - bx
    m["mag_y_body_anom"] = m["mag_y"] - by
    m["mag_z_body_anom"] = m["mag_z"] - bz
    m["mag_x_track_anom"] = -direction_sign * m["mag_x_body_anom"]
    m["mag_y_track_anom"] = m["mag_y_body_anom"]
    m["mag_z_track_anom"] = direction_sign * m["mag_z_body_anom"]
    return m


def align_mag_to_span(segments: list[pd.DataFrame], mag: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for seg in segments:
        out = interpolate_segment_to_mag(seg, mag)
        if not out.empty:
            frames.append(out)
    if not frames:
        return pd.DataFrame()
    aligned = pd.concat(frames, ignore_index=True)
    aligned = aligned[(aligned["s_abs_m"] >= -5) & (aligned["s_abs_m"] <= 1200)].copy()
    return aligned.sort_values(["segment_label", "time"])


def coverage_table(aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in aligned.groupby("segment_label"):
        rows.append(
            {
                "segment_label": label,
                "span_label": g["span_label"].iloc[0],
                "direction": g["direction"].iloc[0],
                "mag_files": "; ".join(sorted({Path(p).name for p in g["mag_file"].unique()})),
                "start_time": g["time"].min().isoformat(sep=" "),
                "end_time": g["time"].max().isoformat(sep=" "),
                "s_min_m": round(float(g["s_abs_m"].min()), 3),
                "s_max_m": round(float(g["s_abs_m"].max()), 3),
                "samples": int(len(g)),
                "mag_total_mean_nT": round(float(g["mag_total"].mean()), 3),
            }
        )
    return pd.DataFrame(rows).sort_values(["s_min_m", "segment_label"])


def choose_full_grid(axis_length: float, step: float = 0.5) -> np.ndarray:
    end = math.floor(min(900.0, max(0.0, axis_length)) / step) * step
    return np.round(np.arange(0.0, end + step / 2, step), 3)


def choose_common_grid(aligned: pd.DataFrame, step: float = 0.5) -> np.ndarray:
    cov = coverage_table(aligned)
    if cov.empty:
        return np.array([])
    # Keep the region covered by at least half of usable segments, robust to short starts/stops.
    starts = cov["s_min_m"].to_numpy(dtype=float)
    ends = cov["s_max_m"].to_numpy(dtype=float)
    start = float(np.nanpercentile(starts, 50))
    end = float(np.nanpercentile(ends, 50))
    if end - start < 200:
        start = float(np.nanpercentile(starts, 25))
        end = float(np.nanpercentile(ends, 75))
    start = math.ceil(max(0.0, start) / step) * step
    end = math.floor(min(900.0, end) / step) * step
    return np.round(np.arange(start, end + step / 2, step), 3)


def interpolate_group_to_grid(group: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    g = group.sort_values("s_abs_m").copy()
    g = g.groupby("s_abs_m", as_index=False).agg(
        {
            "time": "first",
            "lat": "mean",
            "lon": "mean",
            "alt_m": "mean",
            "mag_x": "mean",
            "mag_y": "mean",
            "mag_z": "mean",
            "mag_x_track": "mean",
            "mag_y_track": "mean",
            "mag_z_track": "mean",
            "mag_x_track_anom": "mean",
            "mag_y_track_anom": "mean",
            "mag_z_track_anom": "mean",
            "mag_total": "mean",
            "fix_quality": "median",
            "satellites": "median",
            "hdop": "median",
        }
    )
    x = g["s_abs_m"].to_numpy(dtype=float)
    if len(x) < 2:
        return pd.DataFrame({"distance_m": grid})
    mask = (grid >= x.min()) & (grid <= x.max())
    out = pd.DataFrame({"distance_m": grid})
    out.loc[mask, "time"] = pd.to_datetime(
        np.interp(
            grid[mask],
            x,
            g["time"].astype("int64").to_numpy(dtype=float),
        )
    )
    for col in [
        "lat",
        "lon",
        "alt_m",
        "mag_x",
        "mag_y",
        "mag_z",
        "mag_x_track",
        "mag_y_track",
        "mag_z_track",
        "mag_x_track_anom",
        "mag_y_track_anom",
        "mag_z_track_anom",
        "mag_total",
        "fix_quality",
        "satellites",
        "hdop",
    ]:
        out.loc[mask, col] = np.interp(grid[mask], x, g[col].to_numpy(dtype=float))
    return out


def build_map_csv(
    aligned: pd.DataFrame,
    grid: np.ndarray,
    axis: np.ndarray,
    origin_xy: np.ndarray,
    ref_lat: float,
    ref_lon: float,
) -> pd.DataFrame:
    map_df = pd.DataFrame({"distance_m": grid})
    fit_xy = origin_xy[None, :] + grid[:, None] * axis[None, :]
    fit_lat, fit_lon = local_xy_to_latlon(fit_xy[:, 0], fit_xy[:, 1], ref_lat, ref_lon)
    map_df["track_fit_x_east_m"] = fit_xy[:, 0]
    map_df["track_fit_y_north_m"] = fit_xy[:, 1]
    map_df["track_fit_lat"] = fit_lat
    map_df["track_fit_lon"] = fit_lon
    coord_frames = []
    for label, g in aligned.groupby("segment_label"):
        interp = interpolate_group_to_grid(g, grid)
        coord_frames.append(interp[["distance_m", "lat", "lon", "alt_m"]])
        short = label
        for col in [
            "time",
            "lat",
            "lon",
            "alt_m",
            "mag_x",
            "mag_y",
            "mag_z",
            "mag_x_track",
            "mag_y_track",
            "mag_z_track",
            "mag_x_track_anom",
            "mag_y_track_anom",
            "mag_z_track_anom",
            "mag_total",
            "fix_quality",
            "satellites",
            "hdop",
        ]:
            new_col = f"{short}_{col}"
            map_df[new_col] = interp[col] if col in interp else np.nan
        map_df[f"{short}_sample_count_near_grid"] = (
            g.assign(bin=lambda d: np.round(d["s_abs_m"] * 2) / 2)
            .groupby("bin")
            .size()
            .reindex(grid, fill_value=0)
            .to_numpy()
        )

    coords = pd.concat(coord_frames, ignore_index=True)
    coord_mean = coords.groupby("distance_m", as_index=False).mean(numeric_only=True)
    map_df = map_df.merge(coord_mean, on="distance_m", how="left", suffixes=("", "_mean"))
    for col in ["lat", "lon", "alt_m"]:
        if col in map_df:
            map_df.rename(columns={col: f"map_{col}"}, inplace=True)
    map_df["map_lat"] = map_df["map_lat"].fillna(map_df["track_fit_lat"])
    map_df["map_lon"] = map_df["map_lon"].fillna(map_df["track_fit_lon"])
    mag_total_cols = [c for c in map_df.columns if c.endswith("_mag_total")]
    mag_x_cols = [c for c in map_df.columns if c.endswith("_mag_x")]
    mag_y_cols = [c for c in map_df.columns if c.endswith("_mag_y")]
    mag_z_cols = [c for c in map_df.columns if c.endswith("_mag_z")]
    mag_x_track_cols = [c for c in map_df.columns if c.endswith("_mag_x_track")]
    mag_y_track_cols = [c for c in map_df.columns if c.endswith("_mag_y_track")]
    mag_z_track_cols = [c for c in map_df.columns if c.endswith("_mag_z_track")]
    mag_x_track_anom_cols = [c for c in map_df.columns if c.endswith("_mag_x_track_anom")]
    mag_y_track_anom_cols = [c for c in map_df.columns if c.endswith("_mag_y_track_anom")]
    mag_z_track_anom_cols = [c for c in map_df.columns if c.endswith("_mag_z_track_anom")]
    map_df["map_mag_total_mean_nT"] = map_df[mag_total_cols].mean(axis=1, skipna=True)
    map_df["map_mag_total_std_nT"] = map_df[mag_total_cols].std(axis=1, skipna=True)
    map_df["map_mag_x_mean_nT"] = map_df[mag_x_cols].mean(axis=1, skipna=True)
    map_df["map_mag_y_mean_nT"] = map_df[mag_y_cols].mean(axis=1, skipna=True)
    map_df["map_mag_z_mean_nT"] = map_df[mag_z_cols].mean(axis=1, skipna=True)
    map_df["map_mag_x_track_mean_nT"] = map_df[mag_x_track_cols].mean(axis=1, skipna=True)
    map_df["map_mag_y_track_mean_nT"] = map_df[mag_y_track_cols].mean(axis=1, skipna=True)
    map_df["map_mag_z_track_mean_nT"] = map_df[mag_z_track_cols].mean(axis=1, skipna=True)
    map_df["map_mag_x_track_anom_mean_nT"] = map_df[mag_x_track_anom_cols].mean(axis=1, skipna=True)
    map_df["map_mag_y_track_anom_mean_nT"] = map_df[mag_y_track_anom_cols].mean(axis=1, skipna=True)
    map_df["map_mag_z_track_anom_mean_nT"] = map_df[mag_z_track_anom_cols].mean(axis=1, skipna=True)
    map_df["map_pass_count"] = map_df[mag_total_cols].notna().sum(axis=1)
    ordered = [
        "distance_m",
        "map_lat",
        "map_lon",
        "map_alt_m",
        "track_fit_lat",
        "track_fit_lon",
        "track_fit_x_east_m",
        "track_fit_y_north_m",
        "map_pass_count",
        "map_mag_x_mean_nT",
        "map_mag_y_mean_nT",
        "map_mag_z_mean_nT",
        "map_mag_x_track_mean_nT",
        "map_mag_y_track_mean_nT",
        "map_mag_z_track_mean_nT",
        "map_mag_x_track_anom_mean_nT",
        "map_mag_y_track_anom_mean_nT",
        "map_mag_z_track_anom_mean_nT",
        "map_mag_total_mean_nT",
        "map_mag_total_std_nT",
    ]
    remaining = [c for c in map_df.columns if c not in ordered]
    return map_df[ordered + remaining]


def prepare_global_span_geometry(leap_seconds: int) -> tuple[dict[str, list[SpanFile]], float, float, np.ndarray, np.ndarray, float]:
    prepared = {name: read_dataset_span(name, info, leap_seconds) for name, info in DATASETS.items()}
    all_span_files = [sf for files in prepared.values() for sf in files]
    if not all_span_files:
        raise RuntimeError("No SPAN GPGGA files found")
    lat0, lon0 = add_local_xy(all_span_files)
    axis, origin_xy, axis_length = fit_track_axis(all_span_files)
    return prepared, lat0, lon0, axis, origin_xy, axis_length


def process_dataset(
    name: str,
    info: dict,
    write: bool,
    leap_seconds: int,
    span_files: list[SpanFile] | None = None,
    global_ref_lat: float | None = None,
    global_ref_lon: float | None = None,
    global_axis: np.ndarray | None = None,
    global_origin_xy: np.ndarray | None = None,
    global_axis_length: float | None = None,
) -> dict:
    if span_files is None:
        span_files = read_dataset_span(name, info, leap_seconds)
        lat0, lon0 = add_local_xy(span_files)
        axis, origin_xy, axis_length = fit_track_axis(span_files)
    else:
        lat0 = float(global_ref_lat)
        lon0 = float(global_ref_lon)
        axis = np.asarray(global_axis, dtype=float)
        origin_xy = np.asarray(global_origin_xy, dtype=float)
        axis_length = float(global_axis_length)
    if not span_files:
        raise RuntimeError(f"No SPAN GPGGA files found for {name}")
    segments = build_span_segments(span_files)
    mag = read_dataset_mag(info)
    aligned = align_mag_to_span(segments, mag)
    cov = coverage_table(aligned)
    grid = choose_full_grid(axis_length)
    map_df = build_map_csv(aligned, grid, axis, origin_xy, lat0, lon0) if len(grid) else pd.DataFrame()

    report = {
        "dataset": name,
        "survey_date": info["survey_date"].strftime("%Y-%m-%d"),
        "span_files": [
            {
                "label": sf.label,
                "path": str(sf.path),
                "rows": int(len(sf.df)),
                "start_local": sf.df["time"].min().isoformat(sep=" "),
                "end_local": sf.df["time"].max().isoformat(sep=" "),
                "s_min_m": round(float(sf.df["s_abs_m"].min()), 3),
                "s_max_m": round(float(sf.df["s_abs_m"].max()), 3),
            }
            for sf in span_files
        ],
        "mag_rows": int(len(mag)),
        "aligned_rows": int(len(aligned)),
        "segments": cov.to_dict(orient="records"),
        "reference_lat": lat0,
        "reference_lon": lon0,
        "coordinate_mode": "global_shared_axis_and_origin" if global_axis is not None else "per_dataset_axis_and_origin",
        "track_axis_east_north": [float(axis[0]), float(axis[1])],
        "track_origin_x_east_m": float(origin_xy[0]),
        "track_origin_y_north_m": float(origin_xy[1]),
        "robust_span_axis_length_m": round(float(axis_length), 3),
        "grid_start_m": float(grid[0]) if len(grid) else None,
        "grid_end_m": float(grid[-1]) if len(grid) else None,
        "grid_step_m": 0.5,
        "grid_points": int(len(grid)),
        "gps_to_utc_leap_seconds_subtracted": int(leap_seconds),
        "notes": [
            "GPGGA time field is treated as UTC according to NovAtel/Hexagon OEM7 documentation, then converted to Beijing time by adding 8 hours.",
            "SPAN duplicate root/Converted_On files are de-duplicated by file name and size.",
            "Folder names are not used for pairing; magnetic data are matched to SPAN segments by timestamp.",
            "Track distance is the projection of RTK GPGGA positions onto one PCA-fitted rail axis; direction is only a quality label, not a separate coordinate system.",
            "Body-frame magnetic axes are also converted to a shared track frame: X along increasing distance, Y down, Z right when facing increasing distance.",
            "For vector map features, per-pass body-frame medians are removed before direction rotation; use *_track_anom columns for cross-pass xyz comparison.",
            "Map point coordinates are the mean of SPAN coordinates interpolated at that distance; endpoints without observations are filled from the fitted rail line.",
        ],
    }

    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        prefix = f"magmap_{name.replace('.', '_')}"
        map_path = OUT_DIR / f"{prefix}_0p5m.csv"
        aligned_path = OUT_DIR / f"{prefix}_aligned_samples.csv"
        cov_path = OUT_DIR / f"{prefix}_segments.csv"
        report_path = OUT_DIR / f"{prefix}_report.json"
        map_df.to_csv(map_path, index=False, encoding="utf-8-sig")
        aligned.to_csv(aligned_path, index=False, encoding="utf-8-sig")
        cov.to_csv(cov_path, index=False, encoding="utf-8-sig")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["outputs"] = {
            "map_csv": str(map_path),
            "aligned_samples_csv": str(aligned_path),
            "segments_csv": str(cov_path),
            "report_json": str(report_path),
        }
    return report


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write CSV outputs to data_proc")
    parser.add_argument("--leap-seconds", type=int, default=GPS_UTC_LEAP_SECONDS)
    parser.add_argument("--per-dataset-axis", action="store_true", help="fit a separate rail axis for each date")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = args.out_dir
    if args.per_dataset_axis:
        reports = [process_dataset(name, info, args.write, args.leap_seconds) for name, info in DATASETS.items()]
    else:
        prepared, lat0, lon0, axis, origin_xy, axis_length = prepare_global_span_geometry(args.leap_seconds)
        reports = [
            process_dataset(
                name,
                info,
                args.write,
                args.leap_seconds,
                span_files=prepared[name],
                global_ref_lat=lat0,
                global_ref_lon=lon0,
                global_axis=axis,
                global_origin_xy=origin_xy,
                global_axis_length=axis_length,
            )
            for name, info in DATASETS.items()
        ]
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
