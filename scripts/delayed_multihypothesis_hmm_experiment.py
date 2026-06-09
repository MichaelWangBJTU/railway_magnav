from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anchor_reference_hmm_experiment as arh
import axis_calibrated_hmm_experiment as hmm


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\delayed_multihypothesis_hmm_experiment")
AXIS_VARIANT = "fwd_z_y_x_back_z_y_minusx"
SAMPLE_PERIOD = "4s"
FEATURES = ["total_raw_hp_z"]
WEIGHTS = {"total_raw_hp_z": 1.0}
SIGMA = 1.2
VMAX = 1.2
STEP_M = 0.5


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def likelihood(q: hmm.QuerySegment, ref: dict[str, np.ndarray]) -> np.ndarray:
    return hmm.measurement_loglikelihood(q, ref, FEATURES, WEIGHTS, sigma=SIGMA, robust=True)


def viterbi_window(
    q: hmm.QuerySegment,
    ref: dict[str, np.ndarray],
    ll: np.ndarray,
    start: int,
    stop: int,
    init_prior: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dist = ref["distance_m"]
    n_s = len(dist)
    times = pd.to_datetime(q.time[start:stop])
    ts = times.astype("int64") / 1e9
    direction_sign = 1 if q.direction == "forward" else -1
    n_t = stop - start
    dp = np.full((n_t, n_s), -np.inf, dtype=np.float32)
    prev = np.full((n_t, n_s), -1, dtype=np.int32)
    if init_prior is None:
        dp[0] = ll[start]
    else:
        dp[0] = ll[start] + init_prior.astype(np.float32)

    for kk in range(1, n_t):
        dt = max(0.2, float(ts[kk] - ts[kk - 1]))
        max_step = max(1, int(math.ceil(VMAX * dt / STEP_M)))
        for j in range(n_s):
            if direction_sign > 0:
                lo = max(0, j - max_step)
                hi = j + 1
            else:
                lo = j
                hi = min(n_s, j + max_step + 1)
            cand = dp[kk - 1, lo:hi]
            if cand.size == 0:
                continue
            best_rel = int(np.argmax(cand))
            best_i = lo + best_rel
            moved = abs(j - best_i) * STEP_M
            speed = moved / max(dt, 1e-3)
            smooth_penalty = -0.025 * speed * speed
            dp[kk, j] = cand[best_rel] + smooth_penalty + ll[start + kk, j]
            prev[kk, j] = best_i
    return dp, prev


def backtrace(prev: np.ndarray, final_idx: int) -> np.ndarray:
    path = np.zeros(prev.shape[0], dtype=int)
    path[-1] = int(final_idx)
    for k in range(prev.shape[0] - 1, 0, -1):
        p = int(prev[k, path[k]])
        path[k - 1] = path[k] if p < 0 else p
    return path


def separated_top_indices(scores: np.ndarray, dist: np.ndarray, k: int = 8, sep_m: float = 50.0) -> list[int]:
    order = np.argsort(scores)[::-1]
    out: list[int] = []
    for idx in order:
        if not np.isfinite(scores[idx]):
            continue
        if all(abs(dist[idx] - dist[j]) >= sep_m for j in out):
            out.append(int(idx))
        if len(out) >= k:
            break
    return out


def gaussian_prior(dist: np.ndarray, center: float, sigma_m: float) -> np.ndarray:
    return -0.5 * ((dist - center) / max(sigma_m, 1e-3)) ** 2


def evaluate_path(pred: np.ndarray, truth: np.ndarray, warmup: int) -> dict[str, float]:
    mask = np.isfinite(pred) & np.isfinite(truth)
    mask[:warmup] = False
    err = pred[mask] - truth[mask]
    if err.size == 0:
        return {
            "sample_count": 0,
            "median_abs_error_m": math.nan,
            "mean_abs_error_m": math.nan,
            "rmse_m": math.nan,
            "p90_abs_error_m": math.nan,
            "final_abs_error_m": math.nan,
        }
    return {
        "sample_count": int(err.size),
        "median_abs_error_m": float(np.median(np.abs(err))),
        "mean_abs_error_m": float(np.mean(np.abs(err))),
        "rmse_m": float(np.sqrt(np.mean(err * err))),
        "p90_abs_error_m": float(np.percentile(np.abs(err), 90)),
        "final_abs_error_m": float(abs(err[-1])),
    }


def delayed_multihypothesis(
    q: hmm.QuerySegment,
    ref: dict[str, np.ndarray],
    warmup_samples: int,
    top_k: int = 8,
    sep_m: float = 50.0,
    prior_sigma_m: float = 8.0,
) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
    dist = ref["distance_m"]
    n_t = len(q.time)
    warmup = min(max(5, warmup_samples), max(5, n_t - 5))
    ll = likelihood(q, ref)

    warm_dp, warm_prev = viterbi_window(q, ref, ll, 0, warmup)
    endpoints = separated_top_indices(warm_dp[-1], dist, k=top_k, sep_m=sep_m)
    if not endpoints:
        pred = np.full(n_t, np.nan)
        return pred, {"accepted": 0.0, "reason_code": 1.0}, pd.DataFrame()

    cand_rows = []
    best_score = -np.inf
    best_pred = np.full(n_t, np.nan)
    best_endpoint = endpoints[0]
    for rank, endpoint in enumerate(endpoints, start=1):
        prior = gaussian_prior(dist, dist[endpoint], prior_sigma_m)
        suffix_dp, suffix_prev = viterbi_window(q, ref, ll, warmup, n_t, init_prior=prior)
        suffix_final = int(np.argmax(suffix_dp[-1]))
        warm_path = backtrace(warm_prev, endpoint)
        suffix_path = backtrace(suffix_prev, suffix_final)
        pred_idx = np.concatenate([warm_path, suffix_path])
        pred = dist[pred_idx]
        # Score: warm-up score at selected endpoint + suffix score after delayed initialization.
        total_score = float(warm_dp[-1, endpoint] + np.max(suffix_dp[-1]))
        metrics_after_delay = evaluate_path(pred, q.truth_s, warmup=warmup)
        cand_rows.append(
            {
                "candidate_rank": rank,
                "endpoint_s_m": float(dist[endpoint]),
                "suffix_final_s_m": float(dist[suffix_final]),
                "total_score": total_score,
                **{f"postdelay_{k}": v for k, v in metrics_after_delay.items()},
            }
        )
        if total_score > best_score:
            best_score = total_score
            best_pred = pred
            best_endpoint = endpoint

    cand_df = pd.DataFrame(cand_rows).sort_values("total_score", ascending=False).reset_index(drop=True)
    score_gap = float(cand_df.loc[0, "total_score"] - cand_df.loc[1, "total_score"]) if len(cand_df) > 1 else math.inf
    warm_scores = warm_dp[-1, endpoints]
    warm_gap = float(np.max(warm_scores) - np.partition(warm_scores, -2)[-2]) if len(warm_scores) > 1 else math.inf
    endpoint_spread = float(np.nanmax(cand_df["endpoint_s_m"]) - np.nanmin(cand_df["endpoint_s_m"])) if len(cand_df) else 0.0
    meta = {
        "accepted": 1.0,
        "warmup_samples": float(warmup),
        "warmup_seconds": float((pd.Timestamp(q.time[warmup - 1]) - pd.Timestamp(q.time[0])).total_seconds()),
        "top_k": float(top_k),
        "chosen_endpoint_s_m": float(dist[best_endpoint]),
        "multi_hyp_score_gap": score_gap,
        "warmup_endpoint_score_gap": warm_gap,
        "endpoint_spread_m": endpoint_spread,
        "candidate_count": float(len(cand_df)),
    }
    return best_pred, meta, cand_df


def standard_forward_anchor(q: hmm.QuerySegment, ref: dict[str, np.ndarray]) -> np.ndarray:
    pred, _ = hmm.viterbi_track(
        q,
        ref,
        FEATURES,
        WEIGHTS,
        sigma=SIGMA,
        vmax_mps=VMAX,
        robust=True,
        info_gate=False,
        start_prior="uniform",
    )
    return pred


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in results.groupby("method"):
        rows.append(
            {
                "method": method,
                "segment_count": int(len(g)),
                "median_abs_error_m": float(g["median_abs_error_m"].median()),
                "mean_abs_error_m": float(g["mean_abs_error_m"].mean()),
                "rmse_m": float(g["rmse_m"].mean()),
                "p90_abs_error_m": float(g["p90_abs_error_m"].mean()),
                "final_abs_error_m": float(g["final_abs_error_m"].median()),
                "accepted_count": int(g.get("accepted", pd.Series([1] * len(g))).fillna(1).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["median_abs_error_m", "mean_abs_error_m"])


def apply_rejection(results: pd.DataFrame, gap_thresholds: list[float]) -> pd.DataFrame:
    rows = []
    mh = results[results["method"].str.startswith("DelayedMH")].copy()
    for method, g in mh.groupby("method"):
        for thr in gap_thresholds:
            kept = g[g["multi_hyp_score_gap"] >= thr]
            if kept.empty:
                continue
            rows.append(
                {
                    "method": method,
                    "gap_threshold": thr,
                    "coverage": float(len(kept) / len(g)),
                    "segment_count": int(len(kept)),
                    "kept_segments": ", ".join(kept["segment_label"].tolist()),
                    "median_abs_error_m": float(kept["median_abs_error_m"].median()),
                    "mean_abs_error_m": float(kept["mean_abs_error_m"].mean()),
                    "rmse_m": float(kept["rmse_m"].mean()),
                    "p90_abs_error_m": float(kept["p90_abs_error_m"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["median_abs_error_m", "coverage"], ascending=[True, False])


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5), dpi=180)
    top = summary.sort_values("median_abs_error_m").head(18)
    x = np.arange(len(top))
    ax.bar(x, top["median_abs_error_m"], color="#6a4c93")
    ax.set_xticks(x)
    ax.set_xticklabels(top["method"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Median absolute error / m")
    ax.set_title("Delayed multi-hypothesis HMM initialization")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_notes(summary: pd.DataFrame, rejection: pd.DataFrame, results: pd.DataFrame, path: Path) -> None:
    seg_cols = [
        "method",
        "segment_label",
        "direction",
        "median_abs_error_m",
        "mean_abs_error_m",
        "rmse_m",
        "p90_abs_error_m",
        "warmup_seconds",
        "chosen_endpoint_s_m",
        "multi_hyp_score_gap",
    ]
    lines = [
        "# Delayed Multi-hypothesis HMM Experiment",
        "",
        "Purpose: address no-wheel cold-start ambiguity, especially the `BMAW15230010L_1_seg03` repeated-signature failure.",
        "",
        "Literature link:",
        "",
        "- Recent rail magnetic localization work supports a hybrid sequence-alignment initialization followed by particle-filter tracking.",
        "- HMM map-matching literature also reports persistent initial-phase errors and motivates non-uniform / warm-up initial state probabilities.",
        "- Multiple-candidate map matching keeps several historical candidates alive so early mismatches can be corrected later.",
        "",
        "Implemented method:",
        "",
        "1. Run a warm-up Viterbi pass with uniform initialization for the first W samples.",
        "2. Keep top-K endpoint hypotheses separated by at least 50 m.",
        "3. For each endpoint, continue a suffix HMM with a narrow Gaussian prior around that delayed endpoint.",
        "4. Select the hypothesis with the best combined warm-up + suffix score.",
        "5. Optionally reject if the best-vs-second score gap is too small.",
        "",
        "Aggregate summary:",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Rejection sweep by multi-hypothesis score gap:",
        "",
        rejection.head(20).to_markdown(index=False, floatfmt=".3f") if not rejection.empty else "(no rejection rows)",
        "",
        "Segment-level selected results:",
        "",
        results[seg_cols].sort_values(["method", "segment_label"]).to_markdown(index=False, floatfmt=".3f"),
        "",
        "Interpretation:",
        "",
        "- A useful method should improve `BMAW15230010L_1_seg03` or reject it as ambiguous without hurting the other four segments.",
        "- If delayed multi-hypothesis does not improve over the tuned forward-anchor HMM, the next step should be multi-candidate filtering over time, not only delayed initialization.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    refs, _ = arh.build_candidate_refs()
    ref = refs["forward_only"]
    queries = hmm.read_query_segments(AXIS_VARIANT, SAMPLE_PERIOD)
    rows = []
    cand_rows = []
    traj_rows = []
    warmups = [20, 40, 60, 90]
    for q in queries:
        base_pred = standard_forward_anchor(q, ref)
        base_metrics = evaluate_path(base_pred, q.truth_s, warmup=min(20, max(0, len(q.time) // 10)))
        rows.append(
            {
                "method": "Baseline_forward_anchor_TotalHP_vmax1.2",
                "segment_label": q.label,
                "direction": q.direction,
                "warmup_samples": float(min(20, max(0, len(q.time) // 10))),
                "warmup_seconds": float("nan"),
                "chosen_endpoint_s_m": float("nan"),
                "multi_hyp_score_gap": float("nan"),
                "accepted": 1.0,
                **base_metrics,
            }
        )
        for w in warmups:
            if len(q.time) <= w + 10:
                continue
            pred, meta, cand = delayed_multihypothesis(q, ref, warmup_samples=w, top_k=8, sep_m=50.0, prior_sigma_m=8.0)
            metrics = evaluate_path(pred, q.truth_s, warmup=w)
            method = f"DelayedMH_W{w}_K8_sep50"
            rows.append(
                {
                    "method": method,
                    "segment_label": q.label,
                    "direction": q.direction,
                    **meta,
                    **metrics,
                }
            )
            if not cand.empty:
                cand = cand.copy()
                cand.insert(0, "method", method)
                cand.insert(1, "segment_label", q.label)
                cand.insert(2, "direction", q.direction)
                cand_rows.append(cand)
            keep_idx = np.linspace(0, len(pred) - 1, min(150, len(pred))).round().astype(int)
            for i in keep_idx:
                traj_rows.append(
                    {
                        "method": method,
                        "segment_label": q.label,
                        "direction": q.direction,
                        "time": str(pd.Timestamp(q.time[i])),
                        "truth_s_m": float(q.truth_s[i]),
                        "pred_s_m": float(pred[i]),
                        "error_m": float(pred[i] - q.truth_s[i]),
                    }
                )

    results = pd.DataFrame(rows)
    summary = summarize(results)
    candidates = pd.concat(cand_rows, ignore_index=True) if cand_rows else pd.DataFrame()
    rejection = apply_rejection(results, gap_thresholds=[0, 2, 5, 10, 20, 40, 80])
    traj = pd.DataFrame(traj_rows)
    results.to_csv(OUT_DIR / "delayed_mh_hmm_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "delayed_mh_hmm_summary.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(OUT_DIR / "delayed_mh_hmm_candidates.csv", index=False, encoding="utf-8-sig")
    rejection.to_csv(OUT_DIR / "delayed_mh_hmm_rejection_sweep.csv", index=False, encoding="utf-8-sig")
    traj.to_csv(OUT_DIR / "delayed_mh_hmm_trajectories.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "delayed_mh_hmm_summary.json").write_text(
        json.dumps(
            {
                "summary": summary.to_dict(orient="records"),
                "rejection": rejection.to_dict(orient="records"),
                "results": results.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_summary(summary, OUT_DIR / "delayed_mh_hmm_summary.png")
    write_notes(summary, rejection, results, OUT_DIR / "delayed_mh_hmm_notes.md")
    print(summary.round(3).to_string(index=False))
    print("\nRejection sweep:")
    print(rejection.round(3).head(20).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
