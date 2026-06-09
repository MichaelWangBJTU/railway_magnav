from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import axis_calibrated_full_matching as ac


DATA_PROC = Path(r"C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\data_proc_new")
OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\axis_calibrated_hmm")
STEP_M = 0.5


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def robust_center_scale(x: np.ndarray) -> tuple[float, float]:
    return ac.robust_center_scale(np.asarray(x, dtype=float))


def robust_z(x: np.ndarray, ref_x: np.ndarray | None = None, clip: float = 6.0) -> np.ndarray:
    base = np.asarray(x if ref_x is None else ref_x, dtype=float)
    med, scale = robust_center_scale(base)
    return np.clip((np.asarray(x, dtype=float) - med) / scale, -clip, clip)


def rolling_median(values: np.ndarray, points: int) -> np.ndarray:
    points = max(3, int(points) | 1)
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(points, center=True, min_periods=max(3, points // 4))
        .median()
        .interpolate(limit_direction="both")
        .to_numpy(float)
    )


def highpass_time(values: np.ndarray, sample_period_s: float, window_s: float = 60.0) -> np.ndarray:
    points = max(5, int(round(window_s / max(sample_period_s, 1e-3))) | 1)
    return np.asarray(values, dtype=float) - rolling_median(values, points)


def gaussian_loglike(obs: float, maps: np.ndarray, sigma: float, robust: bool) -> np.ndarray:
    d = obs - maps
    if robust:
        nu = 3.0
        return -0.5 * (nu + 1.0) * np.log1p((d / sigma) ** 2 / nu)
    return -0.5 * (d / sigma) ** 2


def build_reference(axis_variant: str, reference_profile: str) -> dict[str, np.ndarray]:
    dist, ref, _ = ac.build_reference_features(axis_variant, reference_profile=reference_profile)
    out = {"distance_m": dist}
    for key in ["axis_x_hp", "axis_y_hp", "axis_z_hp", "axis_total_hp", "total_raw_hp", "old_x_hp", "old_y_hp"]:
        values = (
            pd.Series(np.asarray(ref[key], dtype=float))
            .interpolate(limit_direction="both")
            .rolling(5, center=True, min_periods=1)
            .median()
            .interpolate(limit_direction="both")
            .to_numpy(float)
        )
        out[key + "_z"] = robust_z(values)
        if not np.isfinite(out[key + "_z"]).all():
            out[key + "_z"] = np.nan_to_num(out[key + "_z"], nan=0.0, posinf=0.0, neginf=0.0)
    return out


@dataclass
class QuerySegment:
    label: str
    direction: str
    time: np.ndarray
    truth_s: np.ndarray
    speed_mps: np.ndarray
    features: dict[str, np.ndarray]


def load_inspvax_speed() -> pd.DataFrame:
    path = Path(r"C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\no_wheel_imu\outputs\inspvax_parsed_summary.csv")
    if not path.exists():
        return pd.DataFrame(columns=["time", "horiz_speed_mps"])
    df = pd.read_csv(path, usecols=["time", "horiz_speed_mps"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["horiz_speed_mps"] = pd.to_numeric(df["horiz_speed_mps"], errors="coerce")
    df = df.dropna().sort_values("time").drop_duplicates("time")
    return df


def interpolate_speed(times: pd.DatetimeIndex, speed_df: pd.DataFrame) -> np.ndarray:
    if speed_df.empty:
        return np.full(len(times), np.nan)
    t_src = speed_df["time"].astype("int64").to_numpy(float) / 1e9
    v_src = speed_df["horiz_speed_mps"].to_numpy(float)
    t_q = times.astype("int64").to_numpy(float) / 1e9
    speed = np.interp(t_q, t_src, v_src, left=np.nan, right=np.nan)
    speed = pd.Series(speed).rolling(5, center=True, min_periods=1).median().interpolate(limit_direction="both").to_numpy(float)
    return np.clip(speed, 0.0, 1.6)


def map_body_5_13_samples(x: np.ndarray, y: np.ndarray, z: np.ndarray, direction: str, axis_variant: str):
    return ac.map_body_5_13(x, y, z, direction, axis_variant)


def read_query_segments(axis_variant: str, sample_period: str = "4s") -> list[QuerySegment]:
    usecols = ["time", "mag_x", "mag_y", "mag_z", "mag_total", "s_abs_m", "segment_label", "direction"]
    df = pd.read_csv(DATA_PROC / "magmap_5_13_aligned_samples.csv", usecols=usecols)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "mag_x", "mag_y", "mag_z", "mag_total", "s_abs_m", "segment_label", "direction"])
    sample_period_s = pd.Timedelta(sample_period).total_seconds()
    speed_df = load_inspvax_speed()
    segments: list[QuerySegment] = []
    for label, part in df.groupby("segment_label", sort=False):
        part = part.sort_values("time").set_index("time")
        direction = str(part["direction"].iloc[0])
        cols = ["mag_x", "mag_y", "mag_z", "mag_total", "s_abs_m"]
        res = part[cols].resample(sample_period).median().interpolate(limit=3, limit_direction="both").dropna()
        if len(res) < 50:
            continue
        duration_s = (res.index[-1] - res.index[0]).total_seconds()
        net_len = abs(float(res["s_abs_m"].iloc[-1]) - float(res["s_abs_m"].iloc[0]))
        if duration_s < 60.0 or net_len < 60.0:
            continue
        x = res["mag_x"].to_numpy(float)
        y = res["mag_y"].to_numpy(float)
        z = res["mag_z"].to_numpy(float)
        total = res["mag_total"].to_numpy(float)
        x_cal, y_cal, z_cal = map_body_5_13_samples(x, y, z, direction, axis_variant)

        direction_sign = 1.0 if direction == "forward" else -1.0
        x_anom = x_cal - np.nanmedian(x_cal)
        y_anom = y_cal - np.nanmedian(y_cal)
        z_anom = z_cal - np.nanmedian(z_cal)
        axis_x_track = -direction_sign * x_anom
        axis_y_track = y_anom
        axis_z_track = direction_sign * z_anom
        total_hp = highpass_time(total, sample_period_s, 80.0)
        features = {
            "axis_x_hp_z": robust_z(highpass_time(axis_x_track, sample_period_s, 60.0)),
            "axis_y_hp_z": robust_z(highpass_time(axis_y_track, sample_period_s, 60.0)),
            "axis_z_hp_z": robust_z(highpass_time(axis_z_track, sample_period_s, 60.0)),
            "axis_total_hp_z": robust_z(total_hp),
            "total_raw_hp_z": robust_z(total_hp),
            # Old y axis is kept as a baseline-like vector feature.
            "old_y_hp_z": robust_z(highpass_time(res["mag_y"].to_numpy(float) - np.nanmedian(res["mag_y"].to_numpy(float)), sample_period_s, 60.0)),
        }
        segments.append(
            QuerySegment(
                label=str(label),
                direction=direction,
                time=res.index.to_numpy(dtype="datetime64[ns]"),
                truth_s=res["s_abs_m"].to_numpy(float),
                speed_mps=interpolate_speed(res.index, speed_df),
                features=features,
            )
        )
    return segments


def measurement_loglikelihood(
    q: QuerySegment,
    ref: dict[str, np.ndarray],
    features: list[str],
    weights: dict[str, float],
    sigma: float,
    robust: bool,
) -> np.ndarray:
    n_t = len(q.truth_s)
    n_s = len(ref["distance_m"])
    ll = np.zeros((n_t, n_s), dtype=np.float32)
    used = 0
    for feat in features:
        if feat not in q.features or feat not in ref:
            continue
        obs = q.features[feat].astype(float)
        maps = ref[feat].astype(float)
        w = float(weights.get(feat, 1.0))
        # Broadcast-friendly loop over time; n_t is only a few hundred.
        for k in range(n_t):
            ll[k] += w * gaussian_loglike(obs[k], maps, sigma=sigma, robust=robust)
        used += 1
    if used == 0:
        raise ValueError("No usable feature")
    ll = np.nan_to_num(ll, nan=-1e6, posinf=-1e6, neginf=-1e6)
    ll -= np.nanmax(ll, axis=1, keepdims=True)
    return ll


def apply_likelihood_uniqueness_gate(
    ll: np.ndarray,
    dist: np.ndarray,
    exclude_m: float = 30.0,
    min_scale: float = 0.12,
    offset: float = 0.03,
    span: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    margins = np.zeros(ll.shape[0], dtype=float)
    for k in range(ll.shape[0]):
        row = ll[k]
        best_i = int(np.nanargmax(row))
        far = np.abs(dist - dist[best_i]) >= exclude_m
        second = np.nanmax(row[far]) if far.any() else np.nanpercentile(row, 99)
        margins[k] = float(row[best_i] - second)
    # Ambiguous measurements are not thrown away, just downweighted.
    scale = np.clip((margins - offset) / span, min_scale, 1.0)
    return (ll.T * scale).T, margins


def viterbi_track(
    q: QuerySegment,
    ref: dict[str, np.ndarray],
    features: list[str],
    weights: dict[str, float],
    sigma: float = 1.2,
    vmax_mps: float = 1.4,
    robust: bool = True,
    info_gate: bool = False,
    gate_min_scale: float = 0.12,
    gate_offset: float = 0.03,
    gate_span: float = 0.20,
    speed_prior: bool = False,
    speed_sigma_mps: float = 0.35,
    speed_weight: float = 0.08,
    start_prior: str = "uniform",
) -> tuple[np.ndarray, dict[str, float]]:
    dist = ref["distance_m"]
    n_s = len(dist)
    ll = measurement_loglikelihood(q, ref, features, weights, sigma=sigma, robust=robust)
    margins = np.full(len(q.time), np.nan)
    if info_gate:
        ll, margins = apply_likelihood_uniqueness_gate(
            ll,
            dist,
            min_scale=gate_min_scale,
            offset=gate_offset,
            span=gate_span,
        )

    times = pd.to_datetime(q.time)
    ts = times.astype("int64") / 1e9
    direction_sign = 1 if q.direction == "forward" else -1
    dp = np.full((len(q.time), n_s), -np.inf, dtype=np.float32)
    prev = np.full((len(q.time), n_s), -1, dtype=np.int32)

    if start_prior == "endpoint_by_direction":
        # This is a practical rail-operation prior: a collected pass starts near
        # one end depending on travel direction. It is not used by default.
        center = dist[0] if direction_sign > 0 else dist[-1]
        prior = -0.5 * ((dist - center) / 80.0) ** 2
        dp[0] = ll[0] + prior.astype(np.float32)
    else:
        dp[0] = ll[0]

    for k in range(1, len(q.time)):
        dt = max(0.2, float(ts[k] - ts[k - 1]))
        max_step = max(1, int(math.ceil(vmax_mps * dt / STEP_M)))
        for j in range(n_s):
            if direction_sign > 0:
                lo = max(0, j - max_step)
                hi = j + 1
            else:
                lo = j
                hi = min(n_s, j + max_step + 1)
            cand = dp[k - 1, lo:hi]
            if cand.size == 0:
                continue
            best_rel = int(np.argmax(cand))
            best_i = lo + best_rel
            moved = abs(j - best_i) * STEP_M
            speed = moved / max(dt, 1e-3)
            smooth_penalty = -0.025 * speed * speed
            speed_penalty = 0.0
            if speed_prior and np.isfinite(q.speed_mps[k]):
                speed_penalty = -speed_weight * ((speed - q.speed_mps[k]) / max(speed_sigma_mps, 1e-3)) ** 2
            dp[k, j] = cand[best_rel] + smooth_penalty + speed_penalty + ll[k, j]
            prev[k, j] = best_i
    path_idx = np.zeros(len(q.time), dtype=int)
    if not np.isfinite(dp[-1]).any():
        # Fall back to the strongest final measurement if the transition model
        # has become numerically disconnected. This should be rare after feature
        # interpolation, and the flag makes it visible in outputs.
        fallback_idx = int(np.nanargmax(ll[-1]))
        return np.full(len(q.time), dist[fallback_idx]), {
            "final_score_margin": math.nan,
            "median_measurement_margin": float(np.nanmedian(margins)),
            "n_time": int(len(q.time)),
            "dp_fallback": 1.0,
        }
    path_idx[-1] = int(np.argmax(dp[-1]))
    for k in range(len(q.time) - 1, 0, -1):
        path_idx[k - 1] = prev[k, path_idx[k]]
        if path_idx[k - 1] < 0:
            path_idx[k - 1] = path_idx[k]
    pred = dist[path_idx]
    meta = {
        "final_score_margin": float(np.nanmax(dp[-1]) - np.nanpercentile(dp[-1], 99)),
        "median_measurement_margin": float(np.nanmedian(margins)),
        "n_time": int(len(q.time)),
        "dp_fallback": 0.0,
    }
    return pred, meta


def evaluate(pred: np.ndarray, truth: np.ndarray, warmup: int = 0) -> dict[str, float]:
    mask = np.isfinite(pred) & np.isfinite(truth)
    if warmup:
        mask[:warmup] = False
    err = pred[mask] - truth[mask]
    if len(err) == 0:
        return {}
    return {
        "sample_count": int(len(err)),
        "mean_error_m": float(np.mean(err)),
        "median_abs_error_m": float(np.median(np.abs(err))),
        "mean_abs_error_m": float(np.mean(np.abs(err))),
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "p75_abs_error_m": float(np.percentile(np.abs(err), 75)),
        "p90_abs_error_m": float(np.percentile(np.abs(err), 90)),
        "final_abs_error_m": float(abs(err[-1])),
    }


def run(axis_variant: str, reference_profile: str, sample_period: str, out_dir: Path) -> None:
    setup_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = build_reference(axis_variant, reference_profile)
    queries = read_query_segments(axis_variant, sample_period)
    configs = [
        {
            "method": "Baseline_TotalHP_Viterbi",
            "features": ["total_raw_hp_z"],
            "weights": {"total_raw_hp_z": 1.0},
            "sigma": 1.2,
            "info_gate": False,
            "start_prior": "uniform",
        },
        {
            "method": "AxisCal_XYHP_Viterbi",
            "features": ["axis_x_hp_z", "axis_y_hp_z"],
            "weights": {"axis_x_hp_z": 1.0, "axis_y_hp_z": 1.0},
            "sigma": 1.35,
            "info_gate": False,
            "start_prior": "uniform",
        },
        {
            "method": "AxisCal_XY_TotalHP_Viterbi",
            "features": ["axis_x_hp_z", "axis_y_hp_z", "axis_total_hp_z"],
            "weights": {"axis_x_hp_z": 0.8, "axis_y_hp_z": 0.8, "axis_total_hp_z": 1.0},
            "sigma": 1.35,
            "info_gate": False,
            "start_prior": "uniform",
        },
        {
            "method": "AxisCal_XY_TotalHP_InfoGate_Viterbi",
            "features": ["axis_x_hp_z", "axis_y_hp_z", "axis_total_hp_z"],
            "weights": {"axis_x_hp_z": 0.8, "axis_y_hp_z": 0.8, "axis_total_hp_z": 1.0},
            "sigma": 1.35,
            "info_gate": True,
            "gate_min_scale": 0.12,
            "gate_offset": 0.03,
            "gate_span": 0.20,
            "start_prior": "uniform",
        },
        {
            "method": "AxisCal_XY_TotalHP_SoftGate_Viterbi",
            "features": ["axis_x_hp_z", "axis_y_hp_z", "axis_total_hp_z"],
            "weights": {"axis_x_hp_z": 0.8, "axis_y_hp_z": 0.8, "axis_total_hp_z": 1.0},
            "sigma": 1.35,
            "info_gate": True,
            "gate_min_scale": 0.45,
            "gate_offset": 0.00,
            "gate_span": 0.18,
            "start_prior": "uniform",
        },
        {
            "method": "AxisCal_XY_TotalHP_MidGate_Viterbi",
            "features": ["axis_x_hp_z", "axis_y_hp_z", "axis_total_hp_z"],
            "weights": {"axis_x_hp_z": 0.8, "axis_y_hp_z": 0.8, "axis_total_hp_z": 1.0},
            "sigma": 1.35,
            "info_gate": True,
            "gate_min_scale": 0.30,
            "gate_offset": 0.02,
            "gate_span": 0.24,
            "start_prior": "uniform",
        },
        {
            "method": "TotalHP_InfoGate_Viterbi",
            "features": ["total_raw_hp_z"],
            "weights": {"total_raw_hp_z": 1.0},
            "sigma": 1.2,
            "info_gate": True,
            "gate_min_scale": 0.12,
            "gate_offset": 0.03,
            "gate_span": 0.20,
            "start_prior": "uniform",
        },
        {
            "method": "EndpointPrior_TotalHP_Viterbi",
            "features": ["total_raw_hp_z"],
            "weights": {"total_raw_hp_z": 1.0},
            "sigma": 1.2,
            "info_gate": False,
            "start_prior": "endpoint_by_direction",
        },
        {
            "method": "EndpointPrior_AxisCal_XY_TotalHP_InfoGate_Viterbi",
            "features": ["axis_x_hp_z", "axis_y_hp_z", "axis_total_hp_z"],
            "weights": {"axis_x_hp_z": 0.8, "axis_y_hp_z": 0.8, "axis_total_hp_z": 1.0},
            "sigma": 1.35,
            "info_gate": True,
            "gate_min_scale": 0.12,
            "gate_offset": 0.03,
            "gate_span": 0.20,
            "start_prior": "endpoint_by_direction",
        },
        {
            "method": "SpeedPrior_TotalHP_Viterbi",
            "features": ["total_raw_hp_z"],
            "weights": {"total_raw_hp_z": 1.0},
            "sigma": 1.2,
            "info_gate": False,
            "speed_prior": True,
            "speed_sigma_mps": 0.35,
            "speed_weight": 0.08,
            "start_prior": "uniform",
        },
        {
            "method": "SpeedPrior_AxisCal_XY_TotalHP_MidGate_Viterbi",
            "features": ["axis_x_hp_z", "axis_y_hp_z", "axis_total_hp_z"],
            "weights": {"axis_x_hp_z": 0.8, "axis_y_hp_z": 0.8, "axis_total_hp_z": 1.0},
            "sigma": 1.35,
            "info_gate": True,
            "gate_min_scale": 0.30,
            "gate_offset": 0.02,
            "gate_span": 0.24,
            "speed_prior": True,
            "speed_sigma_mps": 0.35,
            "speed_weight": 0.08,
            "start_prior": "uniform",
        },
        {
            "method": "SpeedPrior_LightAxis_TotalHP_Viterbi",
            "features": ["axis_x_hp_z", "axis_y_hp_z", "axis_total_hp_z"],
            "weights": {"axis_x_hp_z": 0.25, "axis_y_hp_z": 0.25, "axis_total_hp_z": 1.0},
            "sigma": 1.25,
            "info_gate": False,
            "speed_prior": True,
            "speed_sigma_mps": 0.35,
            "speed_weight": 0.08,
            "start_prior": "uniform",
        },
        {
            "method": "SpeedPrior_LightAxis_TotalHP_SoftGate_Viterbi",
            "features": ["axis_x_hp_z", "axis_y_hp_z", "axis_total_hp_z"],
            "weights": {"axis_x_hp_z": 0.25, "axis_y_hp_z": 0.25, "axis_total_hp_z": 1.0},
            "sigma": 1.25,
            "info_gate": True,
            "gate_min_scale": 0.45,
            "gate_offset": 0.00,
            "gate_span": 0.18,
            "speed_prior": True,
            "speed_sigma_mps": 0.35,
            "speed_weight": 0.08,
            "start_prior": "uniform",
        },
    ]
    rows = []
    traj_rows = []
    for q in queries:
        warmup = min(20, max(0, len(q.time) // 10))
        for cfg in configs:
            pred, meta = viterbi_track(
                q,
                ref,
                cfg["features"],
                cfg["weights"],
                sigma=cfg["sigma"],
                vmax_mps=1.4,
                robust=True,
                info_gate=cfg["info_gate"],
                gate_min_scale=cfg.get("gate_min_scale", 0.12),
                gate_offset=cfg.get("gate_offset", 0.03),
                gate_span=cfg.get("gate_span", 0.20),
                speed_prior=cfg.get("speed_prior", False),
                speed_sigma_mps=cfg.get("speed_sigma_mps", 0.35),
                speed_weight=cfg.get("speed_weight", 0.08),
                start_prior=cfg["start_prior"],
            )
            metrics = evaluate(pred, q.truth_s, warmup=warmup)
            rows.append(
                {
                    "axis_variant": axis_variant,
                    "reference_profile": reference_profile,
                    "sample_period": sample_period,
                    "method": cfg["method"],
                    "segment_label": q.label,
                    "direction": q.direction,
                    **metrics,
                    **meta,
                }
            )
            for i in np.linspace(0, len(pred) - 1, min(250, len(pred))).round().astype(int):
                traj_rows.append(
                    {
                        "method": cfg["method"],
                        "segment_label": q.label,
                        "direction": q.direction,
                        "time": str(pd.Timestamp(q.time[i])),
                        "truth_s_m": float(q.truth_s[i]),
                        "pred_s_m": float(pred[i]),
                        "error_m": float(pred[i] - q.truth_s[i]),
                    }
                )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby("method")
        .agg(
            segment_count=("segment_label", "size"),
            median_abs_error_m=("median_abs_error_m", "median"),
            mean_abs_error_m=("mean_abs_error_m", "mean"),
            rmse_m=("rmse_m", "mean"),
            p75_abs_error_m=("p75_abs_error_m", "mean"),
            p90_abs_error_m=("p90_abs_error_m", "mean"),
            final_abs_error_m=("final_abs_error_m", "median"),
        )
        .reset_index()
        .sort_values("median_abs_error_m")
    )
    traj = pd.DataFrame(traj_rows)
    results.to_csv(out_dir / "axis_calibrated_hmm_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "axis_calibrated_hmm_summary.csv", index=False, encoding="utf-8-sig")
    traj.to_csv(out_dir / "axis_calibrated_hmm_trajectories.csv", index=False, encoding="utf-8-sig")
    (out_dir / "axis_calibrated_hmm_summary.json").write_text(
        json.dumps(
            {
                "axis_variant": axis_variant,
                "reference_profile": reference_profile,
                "sample_period": sample_period,
                "summary": summary.to_dict(orient="records"),
                "results": results.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_summary(summary, out_dir / "axis_calibrated_hmm_summary.png")
    plot_example(traj, out_dir / "axis_calibrated_hmm_example.png")
    print(summary.round(3).to_string(index=False))
    print(f"\nOutputs: {out_dir}")


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=180)
    x = np.arange(len(summary))
    ax.bar(x, summary["median_abs_error_m"], color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"], rotation=20, ha="right")
    ax.set_ylabel("Median absolute error / m")
    ax.set_title("Axis-calibrated no-wheel Viterbi methods")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_example(traj: pd.DataFrame, path: Path) -> None:
    if traj.empty:
        return
    seg = str(traj["segment_label"].value_counts().index[0])
    part = traj[traj["segment_label"] == seg].copy()
    fig, ax = plt.subplots(figsize=(11, 5), dpi=180)
    base = part[part["method"] == part["method"].iloc[0]]
    ax.plot(np.arange(len(base)), base["truth_s_m"], color="black", lw=2, label="SPAN truth")
    for method, g in part.groupby("method"):
        ax.plot(np.arange(len(g)), g["pred_s_m"], lw=1.2, label=method)
    ax.set_title(f"Axis-calibrated HMM example: {seg}")
    ax.set_xlabel("Resampled index")
    ax.set_ylabel("Along-track position / m")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis-variant", default="fwd_z_y_x_back_z_y_minusx")
    parser.add_argument("--reference-profile", default="all", choices=["all", "quality_good"])
    parser.add_argument("--sample-period", default="4s")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    run(args.axis_variant, args.reference_profile, args.sample_period, args.out_dir)


if __name__ == "__main__":
    main()
