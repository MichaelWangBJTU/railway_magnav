from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import axis_calibrated_hmm_experiment as hmm


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\adaptive_speed_scale_experiment")
STEP_M = 0.5


def scaled_query(q: hmm.QuerySegment, scale: float) -> hmm.QuerySegment:
    return replace(q, speed_mps=np.asarray(q.speed_mps, dtype=float) * scale)


def viterbi_point_with_score(
    q: hmm.QuerySegment,
    ref: dict[str, np.ndarray],
    features: list[str],
    weights: dict[str, float],
    sigma: float,
    info_gate: bool,
    gate_min_scale: float = 0.12,
    gate_offset: float = 0.03,
    gate_span: float = 0.20,
    speed_sigma_mps: float = 0.35,
    speed_weight: float = 0.08,
    vmax_mps: float = 1.4,
) -> tuple[np.ndarray, dict[str, float]]:
    dist = ref["distance_m"]
    n_s = len(dist)
    ll = hmm.measurement_loglikelihood(q, ref, features, weights, sigma=sigma, robust=True)
    margins = np.full(len(q.time), np.nan)
    if info_gate:
        ll, margins = hmm.apply_likelihood_uniqueness_gate(
            ll,
            dist,
            min_scale=gate_min_scale,
            offset=gate_offset,
            span=gate_span,
        )
    times = pd.to_datetime(q.time)
    ts = times.astype("int64").to_numpy(float) / 1e9
    direction_sign = 1 if q.direction == "forward" else -1
    dp = np.full((len(q.time), n_s), -np.inf, dtype=np.float32)
    prev = np.full((len(q.time), n_s), -1, dtype=np.int32)
    dp[0] = ll[0]
    for k in range(1, len(q.time)):
        dt = max(0.2, float(ts[k] - ts[k - 1]))
        max_step = max(1, int(math.ceil(vmax_mps * dt / STEP_M)))
        for j in range(n_s):
            if direction_sign > 0:
                lo, hi = max(0, j - max_step), j + 1
            else:
                lo, hi = j, min(n_s, j + max_step + 1)
            cand = dp[k - 1, lo:hi]
            if cand.size == 0:
                continue
            best_rel = int(np.argmax(cand))
            best_i = lo + best_rel
            moved = abs(j - best_i) * STEP_M
            speed = moved / max(dt, 1e-3)
            smooth_penalty = -0.025 * speed * speed
            speed_penalty = 0.0
            if np.isfinite(q.speed_mps[k]):
                speed_penalty = -speed_weight * ((speed - q.speed_mps[k]) / max(speed_sigma_mps, 1e-3)) ** 2
            dp[k, j] = cand[best_rel] + smooth_penalty + speed_penalty + ll[k, j]
            prev[k, j] = best_i
    if not np.isfinite(dp[-1]).any():
        fallback_idx = int(np.nanargmax(ll[-1]))
        return np.full(len(q.time), dist[fallback_idx]), {
            "final_best_score": -np.inf,
            "final_score_margin": math.nan,
            "median_measurement_margin": float(np.nanmedian(margins)),
            "dp_fallback": 1.0,
        }
    path_idx = np.zeros(len(q.time), dtype=int)
    path_idx[-1] = int(np.argmax(dp[-1]))
    for k in range(len(q.time) - 1, 0, -1):
        path_idx[k - 1] = prev[k, path_idx[k]]
        if path_idx[k - 1] < 0:
            path_idx[k - 1] = path_idx[k]
    return dist[path_idx], {
        "final_best_score": float(np.nanmax(dp[-1])),
        "final_score_margin": float(np.nanmax(dp[-1]) - np.nanpercentile(dp[-1], 99)),
        "median_measurement_margin": float(np.nanmedian(margins)),
        "dp_fallback": 0.0,
    }


def summarize_selection(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lambdas = [0.0, 0.5, 2.0, 5.0, 10.0]
    for method, part in results.groupby("method_base"):
        for seg, g in part.groupby("segment_label"):
            oracle = g.loc[g["median_abs_error_m"].idxmin()]
            row = {
                "method_base": method,
                "segment_label": seg,
                "oracle_scale": float(oracle["speed_scale"]),
                "oracle_median_abs_error_m": float(oracle["median_abs_error_m"]),
            }
            for lam in lambdas:
                score = g["final_best_score"] - lam * (np.log(g["speed_scale"].astype(float)) ** 2)
                sel = g.loc[score.idxmax()]
                tag = f"lambda_{lam:g}"
                row[f"{tag}_scale"] = float(sel["speed_scale"])
                row[f"{tag}_median_abs_error_m"] = float(sel["median_abs_error_m"])
                row[f"{tag}_mean_abs_error_m"] = float(sel["mean_abs_error_m"])
                row[f"{tag}_rmse_m"] = float(sel["rmse_m"])
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_selected(sel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, part in sel.groupby("method_base"):
        rows.append(
            {
                "method": method + "_oracle_scale_upperbound",
                "segment_count": len(part),
                "median_abs_error_m": float(part["oracle_median_abs_error_m"].median()),
                "mean_abs_error_m": float(part["oracle_median_abs_error_m"].mean()),
            }
        )
        for lam in [0.0, 0.5, 2.0, 5.0, 10.0]:
            col = f"lambda_{lam:g}_median_abs_error_m"
            rows.append(
                {
                    "method": method + f"_posterior_scale_lambda_{lam:g}",
                    "segment_count": len(part),
                    "median_abs_error_m": float(part[col].median()),
                    "mean_abs_error_m": float(part[col].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values("median_abs_error_m")


def run(out_dir: Path = OUT_DIR) -> None:
    hmm.setup_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = hmm.build_reference("fwd_z_y_x_back_z_y_minusx", "all")
    queries = hmm.read_query_segments("fwd_z_y_x_back_z_y_minusx", "4s")
    scales = [0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.2, 3.0]
    methods = [
        {
            "method_base": "AdaptiveScale_TotalHP",
            "features": ["total_raw_hp_z"],
            "weights": {"total_raw_hp_z": 1.0},
            "sigma": 1.2,
            "info_gate": False,
            "speed_weight": 0.08,
        },
        # Multi-feature scale sweeps are intentionally left out of the first
        # run because the total-field speed-prior method is currently the most
        # stable deployable baseline in RMSE. If scale selection works here,
        # the grid can be expanded later.
    ]
    rows = []
    for q in queries:
        warmup = min(20, max(0, len(q.time) // 10))
        for method in methods:
            for scale in scales:
                qs = scaled_query(q, scale)
                pred, meta = viterbi_point_with_score(
                    qs,
                    ref,
                    method["features"],
                    method["weights"],
                    sigma=method["sigma"],
                    info_gate=method["info_gate"],
                    speed_weight=method["speed_weight"],
                )
                metrics = hmm.evaluate(pred, q.truth_s, warmup=warmup)
                rows.append(
                    {
                        "method_base": method["method_base"],
                        "segment_label": q.label,
                        "direction": q.direction,
                        "speed_scale": scale,
                        **metrics,
                        **meta,
                    }
                )
    results = pd.DataFrame(rows)
    selected = summarize_selection(results)
    selected_summary = aggregate_selected(selected)
    results.to_csv(out_dir / "adaptive_speed_scale_grid_results.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(out_dir / "adaptive_speed_scale_selected_by_segment.csv", index=False, encoding="utf-8-sig")
    selected_summary.to_csv(out_dir / "adaptive_speed_scale_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "adaptive_speed_scale_summary.json").write_text(
        json.dumps(
            {
                "scales": scales,
                "selected_summary": selected_summary.to_dict(orient="records"),
                "selected_by_segment": selected.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_summary(selected_summary, out_dir / "adaptive_speed_scale_summary.png")
    write_notes(results, selected, selected_summary, out_dir / "adaptive_speed_scale_notes.md")
    print(selected_summary.round(3).to_string(index=False))
    print(f"\nOutputs: {out_dir}")


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    view = summary.head(12)
    x = np.arange(len(view))
    ax.bar(x, view["median_abs_error_m"], color="#2E74B5")
    ax.set_xticks(x)
    ax.set_xticklabels(view["method"], rotation=35, ha="right")
    ax.set_ylabel("Segment median of median abs error / m")
    ax.set_title("Adaptive weak-mileage scale selection")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_notes(results: pd.DataFrame, selected: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Adaptive Speed-scale Experiment",
        "",
        "Purpose: test whether weak-mileage scale should be treated as a latent variable instead of trusting raw INSPVAX speed.",
        "",
        "Scale grid:",
        "",
        ", ".join(str(x) for x in sorted(results["speed_scale"].unique())),
        "",
        "Selection rules:",
        "",
        "- `oracle_scale_upperbound`: chooses the scale with the smallest segment median error. This is not deployable and only shows potential.",
        "- `posterior_scale_lambda_*`: chooses the scale with largest final Viterbi score, with optional log-scale regularization. This is deployable in principle because it does not use truth.",
        "",
        "Summary:",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Selected by segment:",
        "",
        selected.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Interpretation:",
        "",
        "- If oracle improves but posterior selection does not, scale adaptivity is useful but the current evidence score is not reliable.",
        "- If both improve, adaptive weak mileage is a strong publishable direction.",
        "- If neither improves, INSPVAX/BESTVEL speed should be used only as a weak transition regularizer, not as a sequence-distance source.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
