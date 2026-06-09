from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import axis_calibrated_hmm_experiment as hmm


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\sequence_hmm_experiment")
STEP_M = 0.5


def row_standardize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.nanmean(x, axis=1, keepdims=True)
    std = np.nanstd(x, axis=1, keepdims=True)
    std = np.where(std < eps, np.nan, std)
    return (x - mean) / std


def vector_standardize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray | None:
    x = np.asarray(x, dtype=float)
    if np.isfinite(x).sum() < max(5, int(0.75 * len(x))):
        return None
    x = pd.Series(x).interpolate(limit_direction="both").to_numpy(float)
    std = float(np.nanstd(x))
    if not np.isfinite(std) or std < eps:
        return None
    return (x - float(np.nanmean(x))) / std


def times_seconds(q: hmm.QuerySegment) -> np.ndarray:
    times = pd.to_datetime(q.time)
    return times.astype("int64").to_numpy(float) / 1e9


def cumulative_speed_distance(q: hmm.QuerySegment, fallback_mps: float = 0.55) -> np.ndarray:
    ts = times_seconds(q)
    dt = np.diff(ts, prepend=ts[0])
    dt = np.clip(dt, 0.0, 10.0)
    speed = np.asarray(q.speed_mps, dtype=float).copy()
    finite = np.isfinite(speed) & (speed >= 0)
    if finite.sum() < max(5, len(speed) // 3):
        speed[:] = fallback_mps
    else:
        med = float(np.nanmedian(speed[finite]))
        if not np.isfinite(med) or med < 0.05:
            med = fallback_mps
        speed[~finite] = med
        speed = pd.Series(speed).rolling(5, center=True, min_periods=1).median().to_numpy(float)
        speed = np.clip(speed, 0.0, 1.6)
    return np.cumsum(speed * dt)


def cumulative_truth_distance(q: hmm.QuerySegment) -> np.ndarray:
    truth = np.asarray(q.truth_s, dtype=float)
    return np.abs(truth - truth[0])


def sequence_scores_for_time(
    q_values: np.ndarray,
    ref_values: np.ndarray,
    ref_dist: np.ndarray,
    state_dist: np.ndarray,
    direction_sign: int,
    rel_distance: np.ndarray,
    k: int,
    min_points: int,
) -> np.ndarray:
    valid_hist = np.flatnonzero(rel_distance <= rel_distance[-1] + 1e-9)
    if len(valid_hist) < min_points:
        return np.full(len(state_dist), np.nan, dtype=float)

    q_win = q_values[k - len(rel_distance) + 1 : k + 1]
    qz = vector_standardize(q_win)
    if qz is None:
        return np.full(len(state_dist), np.nan, dtype=float)

    # Position at history time t if current state is s_j:
    # forward: s_t = s_j - travelled_since_t
    # backward: s_t = s_j + travelled_since_t
    pos = state_dist[:, None] - direction_sign * rel_distance[::-1][None, :]
    flat = pos.reshape(-1)
    vals = np.interp(flat, ref_dist, ref_values, left=np.nan, right=np.nan).reshape(pos.shape)
    finite_ratio = np.isfinite(vals).mean(axis=1)
    vals = pd.DataFrame(vals.T).interpolate(limit_direction="both").to_numpy(float).T
    rz = row_standardize(vals)
    score = np.nanmean(rz * qz[None, :], axis=1)
    score[finite_ratio < 0.75] = np.nan
    return score


def point_loglikelihood(
    q: hmm.QuerySegment,
    ref: dict[str, np.ndarray],
    features: list[str],
    weights: dict[str, float],
    sigma: float,
) -> np.ndarray:
    return hmm.measurement_loglikelihood(q, ref, features, weights, sigma=sigma, robust=True)


def sequence_loglikelihood(
    q: hmm.QuerySegment,
    ref: dict[str, np.ndarray],
    features: list[str],
    weights: dict[str, float],
    window_s: float,
    distance_mode: str,
    seq_weight: float,
    min_points: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    dist = ref["distance_m"]
    n_t = len(q.time)
    n_s = len(dist)
    ts = times_seconds(q)
    cumdist = cumulative_truth_distance(q) if distance_mode == "truth_upperbound" else cumulative_speed_distance(q)
    direction_sign = 1 if q.direction == "forward" else -1
    ll = np.zeros((n_t, n_s), dtype=np.float32)
    margins = np.full(n_t, np.nan)

    for k in range(n_t):
        t0 = ts[k] - window_s
        start = int(np.searchsorted(ts, t0, side="left"))
        if k - start + 1 < min_points:
            continue
        rel = cumdist[start : k + 1]
        rel = rel[-1] - rel
        feature_scores = []
        for feat in features:
            q_values = q.features[feat]
            ref_values = ref[feat]
            scores = sequence_scores_for_time(
                q_values,
                ref_values,
                dist,
                dist,
                direction_sign,
                rel,
                k,
                min_points=min_points,
            )
            if np.isfinite(scores).any():
                feature_scores.append(float(weights.get(feat, 1.0)) * scores)
        if not feature_scores:
            continue
        row = np.nanmean(np.vstack(feature_scores), axis=0)
        if not np.isfinite(row).any():
            continue
        best = float(np.nanmax(row))
        row = seq_weight * (row - best)
        ll[k] = np.nan_to_num(row, nan=-8.0 * seq_weight, posinf=0.0, neginf=-8.0 * seq_weight)
        best_i = int(np.argmax(ll[k]))
        far = np.abs(dist - dist[best_i]) >= 30.0
        if far.any():
            margins[k] = float(ll[k, best_i] - np.nanmax(ll[k, far]))
    return ll, margins


@dataclass
class ViterbiConfig:
    method: str
    features: list[str]
    weights: dict[str, float]
    sigma: float = 1.2
    point_weight: float = 0.4
    seq_weight: float = 1.2
    window_s: float = 80.0
    distance_mode: str = "speed"
    speed_prior: bool = True
    speed_sigma_mps: float = 0.35
    speed_weight: float = 0.08
    vmax_mps: float = 1.4


def combine_ll(q: hmm.QuerySegment, ref: dict[str, np.ndarray], cfg: ViterbiConfig) -> tuple[np.ndarray, dict[str, float]]:
    pll = point_loglikelihood(q, ref, cfg.features, cfg.weights, sigma=cfg.sigma)
    sll, seq_margins = sequence_loglikelihood(
        q,
        ref,
        cfg.features,
        cfg.weights,
        window_s=cfg.window_s,
        distance_mode=cfg.distance_mode,
        seq_weight=cfg.seq_weight,
    )
    ll = cfg.point_weight * pll + sll
    ll -= np.nanmax(ll, axis=1, keepdims=True)
    ll = np.nan_to_num(ll, nan=-1e6, posinf=-1e6, neginf=-1e6)
    meta = {
        "median_sequence_margin": float(np.nanmedian(seq_margins)),
        "sequence_margin_finite_rate": float(np.isfinite(seq_margins).mean()),
    }
    return ll.astype(np.float32), meta


def viterbi_from_ll(
    q: hmm.QuerySegment,
    ref: dict[str, np.ndarray],
    ll: np.ndarray,
    cfg: ViterbiConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    dist = ref["distance_m"]
    n_s = len(dist)
    ts = times_seconds(q)
    direction_sign = 1 if q.direction == "forward" else -1
    dp = np.full((len(q.time), n_s), -np.inf, dtype=np.float32)
    prev = np.full((len(q.time), n_s), -1, dtype=np.int32)
    dp[0] = ll[0]

    for k in range(1, len(q.time)):
        dt = max(0.2, float(ts[k] - ts[k - 1]))
        max_step = max(1, int(math.ceil(cfg.vmax_mps * dt / STEP_M)))
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
            if cfg.speed_prior and np.isfinite(q.speed_mps[k]):
                speed_penalty = -cfg.speed_weight * ((speed - q.speed_mps[k]) / max(cfg.speed_sigma_mps, 1e-3)) ** 2
            dp[k, j] = cand[best_rel] + smooth_penalty + speed_penalty + ll[k, j]
            prev[k, j] = best_i

    if not np.isfinite(dp[-1]).any():
        fallback_idx = int(np.nanargmax(ll[-1]))
        return np.full(len(q.time), dist[fallback_idx]), {"final_score_margin": math.nan, "dp_fallback": 1.0}

    path_idx = np.zeros(len(q.time), dtype=int)
    path_idx[-1] = int(np.argmax(dp[-1]))
    for k in range(len(q.time) - 1, 0, -1):
        path_idx[k - 1] = prev[k, path_idx[k]]
        if path_idx[k - 1] < 0:
            path_idx[k - 1] = path_idx[k]
    meta = {
        "final_score_margin": float(np.nanmax(dp[-1]) - np.nanpercentile(dp[-1], 99)),
        "dp_fallback": 0.0,
    }
    return dist[path_idx], meta


def run(out_dir: Path = OUT_DIR) -> None:
    hmm.setup_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = hmm.build_reference("fwd_z_y_x_back_z_y_minusx", "all")
    queries = hmm.read_query_segments("fwd_z_y_x_back_z_y_minusx", "4s")
    configs = [
        ViterbiConfig(
            method="Seq60Speed_TotalHP_Viterbi",
            features=["total_raw_hp_z"],
            weights={"total_raw_hp_z": 1.0},
            sigma=1.2,
            point_weight=0.35,
            seq_weight=1.0,
            window_s=60.0,
            distance_mode="speed",
        ),
        ViterbiConfig(
            method="Seq100Speed_TotalHP_Viterbi",
            features=["total_raw_hp_z"],
            weights={"total_raw_hp_z": 1.0},
            sigma=1.2,
            point_weight=0.35,
            seq_weight=1.2,
            window_s=100.0,
            distance_mode="speed",
        ),
        ViterbiConfig(
            method="Seq100Speed_AxisXYTotal_Viterbi",
            features=["axis_x_hp_z", "axis_y_hp_z", "axis_total_hp_z"],
            weights={"axis_x_hp_z": 0.6, "axis_y_hp_z": 0.6, "axis_total_hp_z": 1.0},
            sigma=1.35,
            point_weight=0.25,
            seq_weight=1.0,
            window_s=100.0,
            distance_mode="speed",
        ),
        ViterbiConfig(
            method="Seq100TruthUB_TotalHP_Viterbi",
            features=["total_raw_hp_z"],
            weights={"total_raw_hp_z": 1.0},
            sigma=1.2,
            point_weight=0.35,
            seq_weight=1.2,
            window_s=100.0,
            distance_mode="truth_upperbound",
            speed_prior=False,
        ),
        ViterbiConfig(
            method="Seq150TruthUB_TotalHP_Viterbi",
            features=["total_raw_hp_z"],
            weights={"total_raw_hp_z": 1.0},
            sigma=1.2,
            point_weight=0.30,
            seq_weight=1.3,
            window_s=150.0,
            distance_mode="truth_upperbound",
            speed_prior=False,
        ),
    ]

    rows = []
    traj_rows = []
    for q in queries:
        warmup = min(20, max(0, len(q.time) // 10))
        for cfg in configs:
            ll, ll_meta = combine_ll(q, ref, cfg)
            pred, meta = viterbi_from_ll(q, ref, ll, cfg)
            metrics = hmm.evaluate(pred, q.truth_s, warmup=warmup)
            rows.append(
                {
                    "method": cfg.method,
                    "segment_label": q.label,
                    "direction": q.direction,
                    "window_s": cfg.window_s,
                    "distance_mode": cfg.distance_mode,
                    **metrics,
                    **ll_meta,
                    **meta,
                }
            )
            take = np.linspace(0, len(pred) - 1, min(250, len(pred))).round().astype(int)
            for i in take:
                traj_rows.append(
                    {
                        "method": cfg.method,
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
            median_sequence_margin=("median_sequence_margin", "median"),
            sequence_margin_finite_rate=("sequence_margin_finite_rate", "mean"),
        )
        .reset_index()
        .sort_values("median_abs_error_m")
    )
    traj = pd.DataFrame(traj_rows)
    results.to_csv(out_dir / "sequence_hmm_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "sequence_hmm_summary.csv", index=False, encoding="utf-8-sig")
    traj.to_csv(out_dir / "sequence_hmm_trajectories.csv", index=False, encoding="utf-8-sig")
    (out_dir / "sequence_hmm_summary.json").write_text(
        json.dumps({"summary": summary.to_dict(orient="records"), "results": results.to_dict(orient="records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_summary(summary, out_dir / "sequence_hmm_summary.png")
    write_notes(summary, results, out_dir / "sequence_hmm_notes.md")
    print(summary.round(3).to_string(index=False))
    print(f"\nOutputs: {out_dir}")


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=180)
    x = np.arange(len(summary))
    ax.bar(x, summary["median_abs_error_m"], color="#2E74B5")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"], rotation=22, ha="right")
    ax.set_ylabel("Median absolute error / m")
    ax.set_title("Sequence-likelihood no-wheel HMM")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_notes(summary: pd.DataFrame, results: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Sequence-likelihood HMM Experiment",
        "",
        "Purpose: merge recent top-k magnetic sequence alignment ideas into the no-wheel HMM, so each observation is a short magnetic signature rather than a single sample.",
        "",
        "Two distance modes were tested:",
        "",
        "- `speed`: deployable diagnostic using INSPVAX speed to estimate the distance covered inside the sequence window.",
        "- `truth_upperbound`: not deployable; uses SPAN truth distance only to estimate the sequence geometry and reveal the best possible effect if weak mileage were perfect.",
        "",
        "Summary:",
        "",
    ]
    lines.append(summary.to_markdown(index=False, floatfmt=".3f"))
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- If `truth_upperbound` is good but `speed` is poor, the bottleneck is weak mileage / speed quality rather than magnetic uniqueness alone.",
            "- If both are poor, sequence matching itself is not sufficient for this railway section and a different reliability or map-quality route is needed.",
            "- These results must be compared against the existing best deployable methods: `AxisCal_XY_TotalHP_MidGate_Viterbi` median 17.6 m and `SpeedPrior_TotalHP_Viterbi` RMSE about 78.9 m.",
            "",
            "Per-segment results:",
            "",
            results[["method", "segment_label", "median_abs_error_m", "mean_abs_error_m", "rmse_m", "final_abs_error_m"]]
            .sort_values(["segment_label", "median_abs_error_m"])
            .to_markdown(index=False, floatfmt=".3f"),
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
