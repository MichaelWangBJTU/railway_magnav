from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path.home() / "Desktop" / "\u78c1\u5bfc\u822a" / "\u6570\u636e" / "codex_railway_magnav"
PROC_DIR = PROJECT_ROOT / "data_proc_new"
OUT_ROOT = PROJECT_ROOT / "no_wheel_sota"
STEP_M = 0.5


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "outputs": root / "outputs",
        "figures": root / "figures",
        "code": root / "code",
        "reports": root / "reports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def robust_center_scale(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.nanstd(x))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return med, scale


def robust_z(x: np.ndarray, ref_x: np.ndarray | None = None, clip: float = 6.0) -> np.ndarray:
    base = np.asarray(x if ref_x is None else ref_x, dtype=float)
    med, scale = robust_center_scale(base)
    z = (np.asarray(x, dtype=float) - med) / scale
    return np.clip(z, -clip, clip)


def rolling_median(values: np.ndarray, points: int) -> np.ndarray:
    points = max(3, int(points) | 1)
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(points, center=True, min_periods=max(3, points // 4))
        .median()
        .interpolate(limit_direction="both")
        .to_numpy(float)
    )


def highpass(values: np.ndarray, window_points: int) -> np.ndarray:
    return np.asarray(values, dtype=float) - rolling_median(values, window_points)


def build_reference(proc_dir: Path) -> dict[str, np.ndarray]:
    ref = pd.read_csv(proc_dir / "magmap_4_14_fused_0p5m.csv")
    dist = ref["distance_m"].to_numpy(float)
    total = ref["mag_total_smooth_nT"].interpolate(limit_direction="both").to_numpy(float)
    y = ref["mag_y_track_anom_smooth_nT"].interpolate(limit_direction="both").to_numpy(float)
    x = ref["mag_x_track_anom_smooth_nT"].interpolate(limit_direction="both").to_numpy(float)
    z = ref["mag_z_track_anom_smooth_nT"].interpolate(limit_direction="both").to_numpy(float)
    total_hp = highpass(total, int(round(40.0 / STEP_M)))
    y_hp = highpass(y, int(round(40.0 / STEP_M)))
    grad_total = np.gradient(total, dist)
    return {
        "distance_m": dist,
        "total_z": robust_z(total),
        "total_hp_z": robust_z(total_hp),
        "y_hp_z": robust_z(y_hp),
        "x_z": robust_z(x),
        "z_z": robust_z(z),
        "grad_total_z": robust_z(grad_total),
    }


@dataclass
class QuerySegment:
    label: str
    direction: str
    time: np.ndarray
    truth_s: np.ndarray
    features: dict[str, np.ndarray]


def read_query_segments(proc_dir: Path, sample_period: str = "2s") -> list[QuerySegment]:
    usecols = [
        "time",
        "mag_total",
        "mag_y_track_anom",
        "mag_x_track_anom",
        "mag_z_track_anom",
        "s_abs_m",
        "segment_label",
        "direction",
    ]
    df = pd.read_csv(proc_dir / "magmap_5_13_aligned_samples.csv", usecols=usecols)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "mag_total", "s_abs_m", "segment_label", "direction"])
    segments: list[QuerySegment] = []
    for label, part in df.groupby("segment_label", sort=False):
        part = part.sort_values("time").set_index("time")
        direction = str(part["direction"].iloc[0])
        res = part[["mag_total", "mag_y_track_anom", "mag_x_track_anom", "mag_z_track_anom", "s_abs_m"]].resample(sample_period).median()
        res = res.interpolate(limit=3, limit_direction="both").dropna()
        if len(res) < 80:
            continue
        duration_s = (res.index[-1] - res.index[0]).total_seconds()
        net_len = abs(float(res["s_abs_m"].iloc[-1]) - float(res["s_abs_m"].iloc[0]))
        if duration_s < 60.0 or net_len < 60.0:
            continue
        # Keep the experiment close to online operation: use robust per-segment
        # calibration, then compare normalized magnetic patterns to the map.
        total = res["mag_total"].to_numpy(float)
        y = res["mag_y_track_anom"].to_numpy(float)
        x = res["mag_x_track_anom"].to_numpy(float)
        z = res["mag_z_track_anom"].to_numpy(float)
        total_hp = highpass(total, max(11, int(round(60.0 / pd.Timedelta(sample_period).total_seconds())) | 1))
        y_hp = highpass(y, max(11, int(round(60.0 / pd.Timedelta(sample_period).total_seconds())) | 1))
        features = {
            "total_z": robust_z(total),
            "total_hp_z": robust_z(total_hp),
            "y_hp_z": robust_z(y_hp),
            "x_z": robust_z(x),
            "z_z": robust_z(z),
        }
        segments.append(
            QuerySegment(
                label=str(label),
                direction=direction,
                time=res.index.to_numpy(dtype="datetime64[ns]"),
                truth_s=res["s_abs_m"].to_numpy(float),
                features=features,
            )
        )
    return segments


def gaussian_loglike(obs: np.ndarray, maps: np.ndarray, sigma: float, robust: bool = False) -> np.ndarray:
    d = obs - maps
    if robust:
        # Student-t-like loss: gentle for ordinary errors, bounded influence for
        # local magnetic disturbances.
        nu = 3.0
        return -0.5 * (nu + 1.0) * np.log1p((d / sigma) ** 2 / nu)
    return -0.5 * (d / sigma) ** 2


def measurement_loglikelihood(
    q: QuerySegment,
    ref: dict[str, np.ndarray],
    feature_set: list[str],
    weights: dict[str, float],
    sigma: float,
    robust: bool,
) -> np.ndarray:
    n_t = len(q.truth_s)
    n_s = len(ref["distance_m"])
    ll = np.zeros((n_t, n_s), dtype=np.float32)
    used = 0
    for feat in feature_set:
        if feat not in q.features or feat not in ref:
            continue
        obs = q.features[feat].astype(float)
        maps = ref[feat].astype(float)
        w = float(weights.get(feat, 1.0))
        for k in range(n_t):
            ll[k] += w * gaussian_loglike(obs[k], maps, sigma=sigma, robust=robust)
        used += 1
    if used == 0:
        raise ValueError("No usable feature")
    ll -= np.nanmax(ll, axis=1, keepdims=True)
    return ll


def viterbi_track(
    q: QuerySegment,
    ref: dict[str, np.ndarray],
    feature_set: list[str],
    weights: dict[str, float],
    sigma: float = 1.2,
    vmax_mps: float = 1.2,
    robust: bool = False,
    info_gate: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    dist = ref["distance_m"]
    n_s = len(dist)
    ll = measurement_loglikelihood(q, ref, feature_set, weights, sigma=sigma, robust=robust)
    if info_gate:
        # Downweight measurements whose map likelihood has many comparable
        # candidates. This is a simple likelihood-ratio reliability gate, related
        # to robust magnetic PF/LRT ideas in the railway literature.
        sorted_ll = np.sort(ll, axis=1)
        margin = sorted_ll[:, -1] - sorted_ll[:, max(0, -1 - int(round(20.0 / STEP_M)))]
        scale = np.clip((margin - 0.02) / 0.12, 0.15, 1.0)
        ll = (ll.T * scale).T
    times = pd.to_datetime(q.time)
    ts = times.astype("int64") / 1e9
    direction_sign = 1 if q.direction == "forward" else -1

    dp = np.full((len(q.time), n_s), -np.inf, dtype=np.float32)
    prev = np.full((len(q.time), n_s), -1, dtype=np.int32)
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
            if len(cand) == 0:
                continue
            best_rel = int(np.argmax(cand))
            best_i = lo + best_rel
            moved = abs(j - best_i) * STEP_M
            # Mild preference for physically plausible smooth motion, while
            # still allowing stops.
            smooth_penalty = -0.02 * (moved / max(dt, 1e-3)) ** 2
            dp[k, j] = cand[best_rel] + smooth_penalty + ll[k, j]
            prev[k, j] = best_i
    path_idx = np.zeros(len(q.time), dtype=int)
    path_idx[-1] = int(np.argmax(dp[-1]))
    for k in range(len(q.time) - 1, 0, -1):
        path_idx[k - 1] = prev[k, path_idx[k]]
        if path_idx[k - 1] < 0:
            path_idx[k - 1] = path_idx[k]
    pred = dist[path_idx]
    score_margin = float(np.nanmax(dp[-1]) - np.nanpercentile(dp[-1], 99))
    meta = {"score_margin": score_margin, "final_score": float(np.nanmax(dp[-1])), "n_time": int(len(q.time))}
    return pred, dp[-1], meta


def particle_filter_track(
    q: QuerySegment,
    ref: dict[str, np.ndarray],
    feature_set: list[str],
    weights: dict[str, float],
    sigma: float = 1.3,
    n_particles: int = 2500,
    vmax_mps: float = 1.2,
    robust: bool = False,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    dist = ref["distance_m"]
    s_min, s_max = float(dist[0]), float(dist[-1])
    sign = 1.0 if q.direction == "forward" else -1.0
    times = pd.to_datetime(q.time)
    ts = times.astype("int64") / 1e9
    pos = rng.uniform(s_min, s_max, n_particles)
    vel = rng.uniform(0.02, vmax_mps, n_particles)
    w = np.full(n_particles, 1.0 / n_particles)
    pred = np.zeros(len(q.time), dtype=float)
    ref_features = {k: ref[k] for k in feature_set if k in ref}
    for k in range(len(q.time)):
        if k > 0:
            dt = max(0.2, float(ts[k] - ts[k - 1]))
            vel = np.clip(vel + rng.normal(0.0, 0.08 * math.sqrt(dt), n_particles), 0.0, vmax_mps)
            pos = pos + sign * vel * dt + rng.normal(0.0, 0.8 * math.sqrt(dt), n_particles)
            pos = np.clip(pos, s_min, s_max)
        logw = np.zeros(n_particles, dtype=float)
        for feat in feature_set:
            if feat not in q.features or feat not in ref_features:
                continue
            maps = np.interp(pos, dist, ref_features[feat])
            logw += weights.get(feat, 1.0) * gaussian_loglike(q.features[feat][k], maps, sigma, robust)
        logw -= np.nanmax(logw)
        w *= np.exp(logw)
        sw = float(np.sum(w))
        if not np.isfinite(sw) or sw <= 0:
            w.fill(1.0 / n_particles)
        else:
            w /= sw
        pred[k] = float(np.sum(w * pos))
        neff = 1.0 / np.sum(w * w)
        if neff < n_particles * 0.5:
            cdf = np.cumsum(w)
            u0 = rng.random() / n_particles
            us = u0 + np.arange(n_particles) / n_particles
            idx = np.searchsorted(cdf, us)
            pos = pos[idx]
            vel = vel[idx]
            w.fill(1.0 / n_particles)
    return pred, {"n_particles": n_particles}


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
        "p90_abs_error_m": float(np.percentile(np.abs(err), 90)),
        "final_abs_error_m": float(abs(err[-1])),
    }


def run(proc_dir: Path, out_root: Path, sample_period: str = "4s") -> None:
    setup_matplotlib()
    dirs = ensure_dirs(out_root)
    ref = build_reference(proc_dir)
    queries = read_query_segments(proc_dir, sample_period)
    configs = [
        {
            "method": "SOTA2018_PF_total",
            "kind": "pf",
            "features": ["total_z"],
            "weights": {"total_z": 1.0},
            "robust": False,
        },
        {
            "method": "SOTA2018_Viterbi_total",
            "kind": "viterbi",
            "features": ["total_z"],
            "weights": {"total_z": 1.0},
            "robust": False,
            "info_gate": False,
        },
        {
            "method": "Proposed_RobustTotalHP_Viterbi",
            "kind": "viterbi",
            "features": ["total_z", "total_hp_z"],
            "weights": {"total_z": 0.7, "total_hp_z": 1.0},
            "robust": True,
            "info_gate": False,
        },
        {
            "method": "Proposed_RobustMultiFeature_Viterbi",
            "kind": "viterbi",
            "features": ["total_z", "total_hp_z", "y_hp_z"],
            "weights": {"total_z": 0.8, "total_hp_z": 1.1, "y_hp_z": 0.9},
            "robust": True,
            "info_gate": True,
        },
        {
            "method": "Proposed_RobustMultiFeature_PF",
            "kind": "pf",
            "features": ["total_z", "total_hp_z", "y_hp_z"],
            "weights": {"total_z": 0.8, "total_hp_z": 1.1, "y_hp_z": 0.9},
            "robust": True,
        },
    ]
    rows = []
    traj_rows = []
    for q in queries:
        for cfg in configs:
            if cfg["kind"] == "pf":
                pred, meta = particle_filter_track(
                    q,
                    ref,
                    cfg["features"],
                    cfg["weights"],
                    robust=cfg.get("robust", False),
                    vmax_mps=1.4,
                    seed=abs(hash((q.label, cfg["method"]))) % (2**32),
                )
            else:
                pred, _, meta = viterbi_track(
                    q,
                    ref,
                    cfg["features"],
                    cfg["weights"],
                    vmax_mps=1.4,
                    robust=cfg.get("robust", False),
                    info_gate=cfg.get("info_gate", False),
                )
            metrics = evaluate(pred, q.truth_s, warmup=min(20, len(pred) // 10))
            rows.append({"method": cfg["method"], "segment_label": q.label, "direction": q.direction, **metrics, **meta})
            for i in np.linspace(0, len(pred) - 1, min(300, len(pred))).round().astype(int):
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
    trajectories = pd.DataFrame(traj_rows)
    summary = (
        results.groupby("method")
        .agg(
            segment_count=("segment_label", "size"),
            median_abs_error_m=("median_abs_error_m", "median"),
            mean_abs_error_m=("mean_abs_error_m", "mean"),
            rmse_m=("rmse_m", "mean"),
            p90_abs_error_m=("p90_abs_error_m", "mean"),
            final_abs_error_m=("final_abs_error_m", "median"),
        )
        .reset_index()
        .sort_values("median_abs_error_m")
    )
    results.to_csv(dirs["outputs"] / "no_wheel_sota_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(dirs["outputs"] / "no_wheel_sota_summary.csv", index=False, encoding="utf-8-sig")
    trajectories.to_csv(dirs["outputs"] / "no_wheel_sota_trajectories.csv", index=False, encoding="utf-8-sig")
    with (dirs["outputs"] / "no_wheel_sota_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_period": sample_period,
                "summary": summary.to_dict(orient="records"),
                "results": results.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    plot_summary(summary, dirs["figures"] / "no_wheel_sota_method_summary.png")
    plot_example(trajectories, dirs["figures"] / "no_wheel_sota_example_trajectories.png")


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=180)
    x = np.arange(len(summary))
    ax.bar(x, summary["median_abs_error_m"], color="#2b6cb0")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"], rotation=20, ha="right")
    ax.set_ylabel("中位绝对误差 / m")
    ax.set_title("无轮速计 SOTA 复现与改进方法对比")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_example(traj: pd.DataFrame, path: Path) -> None:
    if traj.empty:
        return
    methods = list(traj["method"].unique())
    seg = str(traj["segment_label"].value_counts().index[0])
    part = traj[traj["segment_label"] == seg].copy()
    fig, ax = plt.subplots(figsize=(11, 5), dpi=180)
    base = part[part["method"] == methods[0]]
    ax.plot(np.arange(len(base)), base["truth_s_m"], color="black", lw=2, label="SPAN truth")
    for method in methods:
        m = part[part["method"] == method]
        ax.plot(np.arange(len(m)), m["pred_s_m"], lw=1.3, label=method)
    ax.set_title(f"示例轨迹: {seg}")
    ax.set_xlabel("重采样序号")
    ax.set_ylabel("沿轨位置 / m")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc-dir", type=Path, default=PROC_DIR)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--sample-period", default="4s")
    args = parser.parse_args()
    run(args.proc_dir, args.out_root, args.sample_period)
    print(json.dumps({"out_root": str(args.out_root), "done": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
