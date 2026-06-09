from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anchor_reference_selection_experiment as ars
import axis_calibrated_hmm_experiment as hmm
import constrained_map_alignment_experiment as cma


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\anchor_reference_hmm_experiment")
AXIS_VARIANT = "fwd_z_y_x_back_z_y_minusx"
SAMPLE_PERIOD = "4s"


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def clean_smooth_z(values: np.ndarray) -> np.ndarray:
    arr = (
        pd.Series(np.asarray(values, dtype=float))
        .interpolate(limit_direction="both")
        .rolling(5, center=True, min_periods=1)
        .median()
        .interpolate(limit_direction="both")
        .to_numpy(float)
    )
    z = hmm.robust_z(arr)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def to_hmm_reference(ref: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {"distance_m": ref["distance_m"]}
    out["axis_x_hp_z"] = clean_smooth_z(ref["axis_x_hp"])
    out["axis_y_hp_z"] = clean_smooth_z(ref["axis_y_hp"])
    out["axis_z_hp_z"] = clean_smooth_z(ref["axis_z_hp"])
    out["axis_total_hp_z"] = clean_smooth_z(ref["axis_total_hp"])
    out["total_raw_hp_z"] = clean_smooth_z(ref["total_hp"])
    # Keep compatibility with diagnostics that expect this legacy key.
    out["old_y_hp_z"] = out["axis_y_hp_z"].copy()
    return out


def build_candidate_refs() -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[str]]]:
    dist4, passes4 = cma.load_passes("4_14")
    by_seg = ars.pass_by_label(passes4)
    sets = ars.candidate_sets(passes4)
    keep = [
        "all_raw",
        "backward_only",
        "quality_good_exclude_bad",
        "top6_lopo_identity",
        "forward_only",
    ]
    refs: dict[str, dict[str, np.ndarray]] = {}
    for name in keep:
        labels = sets[name]
        refs[name] = to_hmm_reference(ars.build_ref(dist4, [by_seg[seg] for seg in labels]))
    return refs, {name: sets[name] for name in keep}


def configs() -> list[dict]:
    return [
        {
            "method": "TotalHP_Viterbi",
            "features": ["total_raw_hp_z"],
            "weights": {"total_raw_hp_z": 1.0},
            "sigma": 1.2,
            "info_gate": False,
            "start_prior": "uniform",
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
    ]


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["reference_candidate", "method"])
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
    best = summary.sort_values("median_abs_error_m").head(18).copy()
    labels = best["reference_candidate"] + "\n" + best["method"]
    x = np.arange(len(best))
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=180)
    ax.bar(x, best["median_abs_error_m"], color="#2a6fbb")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Median absolute error / m")
    ax.set_title("Anchor-reference HMM candidates")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_best_trajectory(traj: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    if traj.empty or summary.empty:
        return
    best = summary.iloc[0]
    part = traj[
        (traj["reference_candidate"] == best["reference_candidate"])
        & (traj["method"] == best["method"])
    ].copy()
    if part.empty:
        return
    seg = str(part["segment_label"].value_counts().index[0])
    part = part[part["segment_label"] == seg]
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=180)
    ax.plot(np.arange(len(part)), part["truth_s_m"], color="black", lw=2.0, label="SPAN truth")
    ax.plot(np.arange(len(part)), part["pred_s_m"], color="#d62728", lw=1.4, label="HMM estimate")
    ax.set_title(f"Best anchor-HMM example: {best['reference_candidate']} / {best['method']} / {seg}")
    ax.set_xlabel("Resampled index")
    ax.set_ylabel("Along-track position / m")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_notes(summary: pd.DataFrame, candidate_sets: dict[str, list[str]], path: Path) -> None:
    best = summary.sort_values(["median_abs_error_m", "mean_abs_error_m"]).head(12)
    lines = [
        "# Anchor Reference HMM Experiment",
        "",
        "Purpose: verify whether the cleaner 4.14 anchor maps from the map-quality diagnostic also improve no-wheel online-style localization on 5.13.",
        "",
        "HMM/Viterbi setup:",
        "",
        "- State: along-track position on the 0.5 m reference grid.",
        "- Observation: total-field high-pass only, or axis-calibrated X/Y plus total-field high-pass.",
        "- Transition: monotonic motion according to the segment direction, limited by `vmax=1.4 m/s`.",
        "- Optional speed prior: IMU/INSPVAX horizontal speed weakly penalizes implausible step speeds, but no wheel odometer is used.",
        "- Optional information gate: downweights observations whose best magnetic match is not unique relative to alternatives at least 30 m away.",
        "",
        "Reference candidates:",
        "",
    ]
    for name, labels in candidate_sets.items():
        lines.append(f"- `{name}`: {', '.join(labels)}")
    lines += [
        "",
        "Best candidate-method combinations:",
        "",
        best.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Interpretation:",
        "",
        "- If `backward_only` remains best here, the result supports quality/direction-aware map construction instead of naive all-pass averaging.",
        "- If a candidate improves DTW map quality but not HMM error, then the remaining failure is online sequence ambiguity rather than static map repeatability.",
        "- This experiment is deployable-style because Viterbi uses only the observation sequence and weak IMU speed prior, not the 5.13 true position during estimation.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    refs, candidate_sets = build_candidate_refs()
    queries = hmm.read_query_segments(AXIS_VARIANT, SAMPLE_PERIOD)
    rows = []
    traj_rows = []
    for ref_name, ref in refs.items():
        for q in queries:
            warmup = min(20, max(0, len(q.time) // 10))
            for cfg in configs():
                pred, meta = hmm.viterbi_track(
                    q,
                    ref,
                    cfg["features"],
                    cfg["weights"],
                    sigma=cfg["sigma"],
                    vmax_mps=1.4,
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
                        "reference_candidate": ref_name,
                        "method": cfg["method"],
                        "segment_label": q.label,
                        "direction": q.direction,
                        **metrics,
                        **meta,
                    }
                )
                keep_idx = np.linspace(0, len(pred) - 1, min(250, len(pred))).round().astype(int)
                for i in keep_idx:
                    traj_rows.append(
                        {
                            "reference_candidate": ref_name,
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
    results.to_csv(OUT_DIR / "anchor_reference_hmm_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "anchor_reference_hmm_summary.csv", index=False, encoding="utf-8-sig")
    traj.to_csv(OUT_DIR / "anchor_reference_hmm_trajectories.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "anchor_reference_hmm_summary.json").write_text(
        json.dumps(
            {
                "candidate_sets": candidate_sets,
                "summary": summary.to_dict(orient="records"),
                "results": results.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_summary(summary, OUT_DIR / "anchor_reference_hmm_summary.png")
    plot_best_trajectory(traj, summary, OUT_DIR / "anchor_reference_hmm_best_example.png")
    write_notes(summary, candidate_sets, OUT_DIR / "anchor_reference_hmm_notes.md")
    print(summary.round(3).head(20).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
