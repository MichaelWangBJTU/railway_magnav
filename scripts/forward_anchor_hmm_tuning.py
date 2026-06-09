from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anchor_reference_hmm_experiment as arh
import axis_calibrated_hmm_experiment as hmm


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\forward_anchor_hmm_tuning")
AXIS_VARIANT = "fwd_z_y_x_back_z_y_minusx"
SAMPLE_PERIOD = "4s"


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def candidate_configs() -> list[dict]:
    cfgs = []
    for vmax in [0.8, 1.0, 1.2, 1.4, 1.6]:
        for start_prior in ["uniform", "endpoint_by_direction"]:
            cfgs.append(
                {
                    "method": f"TotalHP_vmax{vmax:g}_{start_prior}",
                    "features": ["total_raw_hp_z"],
                    "weights": {"total_raw_hp_z": 1.0},
                    "sigma": 1.2,
                    "vmax_mps": vmax,
                    "info_gate": False,
                    "speed_prior": False,
                    "start_prior": start_prior,
                }
            )
    for vmax in [1.0, 1.2, 1.4]:
        cfgs.append(
            {
                "method": f"TotalHP_InfoGate_vmax{vmax:g}",
                "features": ["total_raw_hp_z"],
                "weights": {"total_raw_hp_z": 1.0},
                "sigma": 1.2,
                "vmax_mps": vmax,
                "info_gate": True,
                "gate_min_scale": 0.20,
                "gate_offset": 0.03,
                "gate_span": 0.20,
                "speed_prior": False,
                "start_prior": "uniform",
            }
        )
        cfgs.append(
            {
                "method": f"TotalHP_SpeedPrior_vmax{vmax:g}",
                "features": ["total_raw_hp_z"],
                "weights": {"total_raw_hp_z": 1.0},
                "sigma": 1.2,
                "vmax_mps": vmax,
                "info_gate": False,
                "speed_prior": True,
                "speed_sigma_mps": 0.35,
                "speed_weight": 0.08,
                "start_prior": "uniform",
            }
        )
    return cfgs


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
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
        .sort_values(["median_abs_error_m", "mean_abs_error_m"])
    )


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    best = summary.sort_values("median_abs_error_m").head(16)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=180)
    x = np.arange(len(best))
    ax.bar(x, best["median_abs_error_m"], color="#1b7f5a")
    ax.set_xticks(x)
    ax.set_xticklabels(best["method"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Median absolute error / m")
    ax.set_title("Forward-anchor TotalHP HMM tuning")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_notes(summary: pd.DataFrame, results: pd.DataFrame, path: Path) -> None:
    best = summary.head(10)
    seg_pivot = (
        results.sort_values("median_abs_error_m")
        .groupby("segment_label")
        .head(5)
        .loc[:, ["segment_label", "direction", "method", "median_abs_error_m", "mean_abs_error_m", "p90_abs_error_m"]]
    )
    lines = [
        "# Forward Anchor HMM Tuning",
        "",
        "Purpose: tune the currently strongest deployable setting, `forward_only` reference map with total-field high-pass HMM.",
        "",
        "Parameters tested:",
        "",
        "- `vmax`: maximum allowed along-track speed in the HMM transition model.",
        "- `endpoint_by_direction`: soft start prior near the route endpoint implied by direction. This is a rail-operation prior, not a truth-position prior.",
        "- `InfoGate`: downweights ambiguous single-point magnetic observations.",
        "- `SpeedPrior`: weak prior from INSPVAX horizontal speed; no wheel speed is used.",
        "",
        "Best settings:",
        "",
        best.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Best settings by segment:",
        "",
        seg_pivot.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Interpretation:",
        "",
        "- If low `vmax` improves errors, earlier failures were partly caused by jumps to repeated magnetic signatures.",
        "- If endpoint prior improves only some segments, it should be treated as an optional operational prior rather than a universal algorithmic gain.",
        "- If SpeedPrior worsens results, the current INSPVAX speed scale is not reliable enough without per-pass calibration.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    refs, _ = arh.build_candidate_refs()
    ref = refs["forward_only"]
    queries = hmm.read_query_segments(AXIS_VARIANT, SAMPLE_PERIOD)
    rows = []
    traj_rows = []
    for q in queries:
        warmup = min(20, max(0, len(q.time) // 10))
        for cfg in candidate_configs():
            pred, meta = hmm.viterbi_track(
                q,
                ref,
                cfg["features"],
                cfg["weights"],
                sigma=cfg["sigma"],
                vmax_mps=cfg["vmax_mps"],
                robust=True,
                info_gate=cfg.get("info_gate", False),
                gate_min_scale=cfg.get("gate_min_scale", 0.12),
                gate_offset=cfg.get("gate_offset", 0.03),
                gate_span=cfg.get("gate_span", 0.20),
                speed_prior=cfg.get("speed_prior", False),
                speed_sigma_mps=cfg.get("speed_sigma_mps", 0.35),
                speed_weight=cfg.get("speed_weight", 0.08),
                start_prior=cfg.get("start_prior", "uniform"),
            )
            metrics = hmm.evaluate(pred, q.truth_s, warmup=warmup)
            rows.append(
                {
                    "method": cfg["method"],
                    "segment_label": q.label,
                    "direction": q.direction,
                    "vmax_mps": cfg["vmax_mps"],
                    "info_gate": int(cfg.get("info_gate", False)),
                    "speed_prior": int(cfg.get("speed_prior", False)),
                    "start_prior": cfg.get("start_prior", "uniform"),
                    **metrics,
                    **meta,
                }
            )
            keep_idx = np.linspace(0, len(pred) - 1, min(180, len(pred))).round().astype(int)
            for i in keep_idx:
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
    summary = summarize(results)
    traj = pd.DataFrame(traj_rows)
    results.to_csv(OUT_DIR / "forward_anchor_hmm_tuning_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "forward_anchor_hmm_tuning_summary.csv", index=False, encoding="utf-8-sig")
    traj.to_csv(OUT_DIR / "forward_anchor_hmm_tuning_trajectories.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "forward_anchor_hmm_tuning_summary.json").write_text(
        json.dumps({"summary": summary.to_dict(orient="records"), "results": results.to_dict(orient="records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_summary(summary, OUT_DIR / "forward_anchor_hmm_tuning_summary.png")
    write_notes(summary, results, OUT_DIR / "forward_anchor_hmm_tuning_notes.md")
    print(summary.round(3).head(16).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
