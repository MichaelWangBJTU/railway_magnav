from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anchor_reference_hmm_experiment as arh
import axis_calibrated_hmm_experiment as hmm
import delayed_multihypothesis_hmm_experiment as dmh


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\robust_candidate_scoring_experiment")
AXIS_VARIANT = "fwd_z_y_x_back_z_y_minusx"
SAMPLE_PERIOD = "4s"


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def candidate_paths(q: hmm.QuerySegment, ref: dict[str, np.ndarray], warmup: int) -> list[dict]:
    dist = ref["distance_m"]
    n_t = len(q.time)
    warmup = min(max(5, warmup), max(5, n_t - 5))
    ll = dmh.likelihood(q, ref)
    warm_dp, warm_prev = dmh.viterbi_window(q, ref, ll, 0, warmup)
    endpoints = dmh.separated_top_indices(warm_dp[-1], dist, k=8, sep_m=50.0)
    out = []
    for endpoint in endpoints:
        prior = dmh.gaussian_prior(dist, dist[endpoint], sigma_m=8.0)
        suffix_dp, suffix_prev = dmh.viterbi_window(q, ref, ll, warmup, n_t, init_prior=prior)
        suffix_final = int(np.argmax(suffix_dp[-1]))
        warm_path = dmh.backtrace(warm_prev, endpoint)
        suffix_path = dmh.backtrace(suffix_prev, suffix_final)
        idx_path = np.concatenate([warm_path, suffix_path])
        pred = dist[idx_path]
        path_ll = ll[np.arange(n_t), idx_path]
        sorted_ll = np.sort(path_ll)
        trim20 = sorted_ll[int(0.2 * len(sorted_ll)) :] if len(sorted_ll) else sorted_ll
        trim35 = sorted_ll[int(0.35 * len(sorted_ll)) :] if len(sorted_ll) else sorted_ll
        total_score = float(warm_dp[-1, endpoint] + np.max(suffix_dp[-1]))
        metrics = dmh.evaluate_path(pred, q.truth_s, warmup=warmup)
        out.append(
            {
                "pred": pred,
                "endpoint_s_m": float(dist[endpoint]),
                "suffix_final_s_m": float(dist[suffix_final]),
                "total_score": total_score,
                "total_score_per_sample": float(total_score / max(n_t, 1)),
                "path_ll_mean": float(np.mean(path_ll)),
                "path_ll_median": float(np.median(path_ll)),
                "path_ll_p10": float(np.percentile(path_ll, 10)),
                "path_ll_trim20_mean": float(np.mean(trim20)),
                "path_ll_trim35_mean": float(np.mean(trim35)),
                "progress_m": float(abs(pred[-1] - pred[0])),
                **{f"candidate_{k}": v for k, v in metrics.items()},
            }
        )
    return out


def score_candidate(c: dict, mode: str) -> float:
    if mode == "sum":
        return float(c["total_score"])
    if mode == "per_sample_sum":
        return float(c["total_score_per_sample"])
    if mode == "median_ll":
        return float(c["path_ll_median"])
    if mode == "trim20_ll":
        return float(c["path_ll_trim20_mean"])
    if mode == "trim35_ll":
        return float(c["path_ll_trim35_mean"])
    if mode == "p10_ll":
        return float(c["path_ll_p10"])
    if mode == "robust_hybrid":
        return float(0.35 * c["total_score_per_sample"] + 0.65 * c["path_ll_trim20_mean"])
    raise ValueError(mode)


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
            }
        )
    return pd.DataFrame(rows).sort_values(["median_abs_error_m", "mean_abs_error_m"])


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    top = summary.head(20)
    fig, ax = plt.subplots(figsize=(12, 5.4), dpi=180)
    x = np.arange(len(top))
    ax.bar(x, top["median_abs_error_m"], color="#c05252")
    ax.set_xticks(x)
    ax.set_xticklabels(top["method"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Median absolute error / m")
    ax.set_title("Robust scoring for delayed multi-hypothesis candidates")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_notes(summary: pd.DataFrame, results: pd.DataFrame, candidates: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Robust Candidate Scoring Experiment",
        "",
        "Purpose: the delayed multi-hypothesis experiment showed that `BMAW15230010L_1_seg03` has near-correct candidates, but cumulative likelihood ranks them too low.",
        "",
        "Scoring modes:",
        "",
        "- `sum`: original cumulative Viterbi score.",
        "- `median_ll`: median measurement log-likelihood along the candidate path.",
        "- `trim20_ll`: mean path log-likelihood after discarding the worst 20% samples.",
        "- `trim35_ll`: mean path log-likelihood after discarding the worst 35% samples.",
        "- `p10_ll`: 10th percentile path log-likelihood, a conservative worst-tail score.",
        "- `robust_hybrid`: weighted combination of per-sample cumulative score and trimmed mean score.",
        "",
        "Aggregate summary:",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Segment-level selected results:",
        "",
        results.sort_values(["method", "segment_label"]).to_markdown(index=False, floatfmt=".3f"),
        "",
        "`1_seg03` candidate table:",
        "",
        candidates[candidates["segment_label"] == "BMAW15230010L_1_seg03"].sort_values(["warmup_samples", "candidate_median_abs_error_m"]).head(30).to_markdown(index=False, floatfmt=".3f"),
        "",
        "Interpretation:",
        "",
        "- If robust scoring improves `1_seg03` without damaging other segments, the next method can be a robust multi-hypothesis HMM.",
        "- If robust scoring selects implausible candidates in other segments, use it only as an integrity diagnostic or combine it with progress constraints.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    refs, _ = arh.build_candidate_refs()
    ref = refs["forward_only"]
    queries = hmm.read_query_segments(AXIS_VARIANT, SAMPLE_PERIOD)
    modes = ["sum", "median_ll", "trim20_ll", "trim35_ll", "p10_ll", "robust_hybrid"]
    warmups = [20, 40, 60, 90]
    rows = []
    cand_rows = []
    for q in queries:
        for w in warmups:
            if len(q.time) <= w + 10:
                continue
            cands = candidate_paths(q, ref, w)
            for rank, c in enumerate(sorted(cands, key=lambda x: x["total_score"], reverse=True), start=1):
                row = {k: v for k, v in c.items() if k != "pred"}
                row.update({"segment_label": q.label, "direction": q.direction, "warmup_samples": w, "sum_rank": rank})
                cand_rows.append(row)
            for mode in modes:
                if not cands:
                    continue
                best = max(cands, key=lambda c: score_candidate(c, mode))
                pred = best["pred"]
                metrics = dmh.evaluate_path(pred, q.truth_s, warmup=w)
                rows.append(
                    {
                        "method": f"RobustScore_{mode}_W{w}",
                        "score_mode": mode,
                        "segment_label": q.label,
                        "direction": q.direction,
                        "warmup_samples": w,
                        "chosen_endpoint_s_m": best["endpoint_s_m"],
                        "chosen_final_s_m": best["suffix_final_s_m"],
                        "selection_score": score_candidate(best, mode),
                        "candidate_sum_score": best["total_score"],
                        "candidate_path_ll_median": best["path_ll_median"],
                        "candidate_path_ll_trim20": best["path_ll_trim20_mean"],
                        **metrics,
                    }
                )
    results = pd.DataFrame(rows)
    candidates = pd.DataFrame(cand_rows)
    summary = summarize(results)
    results.to_csv(OUT_DIR / "robust_candidate_scoring_results.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(OUT_DIR / "robust_candidate_scoring_candidates.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "robust_candidate_scoring_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "robust_candidate_scoring_summary.json").write_text(
        json.dumps(
            {
                "summary": summary.to_dict(orient="records"),
                "results": results.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_summary(summary, OUT_DIR / "robust_candidate_scoring_summary.png")
    write_notes(summary, results, candidates, OUT_DIR / "robust_candidate_scoring_notes.md")
    print(summary.round(3).head(20).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
