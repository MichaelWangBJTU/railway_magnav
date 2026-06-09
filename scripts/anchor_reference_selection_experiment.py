from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import constrained_map_alignment_experiment as cma
import distance_warp_diagnostic as dwd


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\anchor_reference_selection_experiment")


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def build_ref(dist: np.ndarray, passes: list[cma.PassData]) -> dict[str, np.ndarray]:
    arrays = [{feat: p.features[feat] for feat in cma.FEATURE_NAMES} for p in passes]
    return cma.build_reference_from_arrays(dist, arrays)


def pass_by_label(passes: list[cma.PassData]) -> dict[str, cma.PassData]:
    return {p.segment: p for p in passes}


def candidate_sets(passes4: list[cma.PassData]) -> dict[str, list[str]]:
    by_seg = pass_by_label(passes4)
    forward = [p.segment for p in passes4 if p.direction == "forward"]
    backward = [p.segment for p in passes4 if p.direction == "backward"]
    # Same-day LOPO total-field identity ranking from distance_warp_lopo_4_14.
    top4_lopo = [
        "BMAW15230010L_1_seg01",
        "BMAW15230010L_3_seg03",
        "BMAW15230010L_5_seg01",
        "BMAW15230010L_1_seg04",
    ]
    top6_lopo = top4_lopo + [
        "BMAW15230010L_3_seg01",
        "BMAW15230010L_2_seg01",
    ]
    quality_good = [p.segment for p in passes4 if p.segment not in {"BMAW15230010L_1_seg02", "BMAW15230010L_1_seg03"}]
    selective_accepted = [
        "BMAW15230010L_5_seg01",
        "BMAW15230010L_3_seg03",
        "BMAW15230010L_1_seg04",
        "BMAW15230010L_1_seg02",
    ]
    # Keep only labels that exist in the current processed map.
    raw = {
        "all_raw": list(by_seg),
        "quality_good_exclude_bad": quality_good,
        "forward_only": forward,
        "backward_only": backward,
        "top4_lopo_identity": top4_lopo,
        "top6_lopo_identity": top6_lopo,
        "selective_gate_accepted_raw": selective_accepted,
    }
    return {k: [seg for seg in v if seg in by_seg] for k, v in raw.items()}


def identity_feature_corr(q: cma.PassData, ref: dict[str, np.ndarray], feat: str) -> float:
    qz = dwd.z_valid(q.features[feat])
    rz = dwd.z_valid(ref[feat])
    mask = np.isfinite(qz) & np.isfinite(rz)
    if mask.sum() < 20:
        return float("nan")
    return float(np.corrcoef(qz[mask], rz[mask])[0, 1])


def eval_candidate(
    name: str,
    ref: dict[str, np.ndarray],
    selected: list[cma.PassData],
    eval_passes: list[cma.PassData],
    eval_tag: str,
) -> pd.DataFrame:
    rows = []
    selected_labels = {p.segment for p in selected}
    for p in eval_passes:
        total = cma.eval_against_ref(p, ref, f"{name}_{eval_tag}")
        rows.append(
            {
                "candidate": name,
                "eval_tag": eval_tag,
                "segment_label": p.segment,
                "direction": p.direction,
                "date_tag": p.date_tag,
                "in_reference": int(p.segment in selected_labels and p.date_tag == "4_14"),
                "axis_x_identity_corr": identity_feature_corr(p, ref, "axis_x_hp"),
                "axis_y_identity_corr": identity_feature_corr(p, ref, "axis_y_hp"),
                "axis_z_identity_corr": identity_feature_corr(p, ref, "axis_z_hp"),
                **{k: v for k, v in total.items() if k != "eval_ref"},
            }
        )
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (cand, tag), g in rows.groupby(["candidate", "eval_tag"]):
        out.append(
            {
                "candidate": cand,
                "eval_tag": tag,
                "segment_count": int(len(g)),
                "median_identity_corr": float(g["identity_corr"].median()),
                "mean_identity_corr": float(g["identity_corr"].mean()),
                "median_band60_corr": float(g["band60_dtw_corr"].median()),
                "mean_band60_corr": float(g["band60_dtw_corr"].mean()),
                "median_band60_rms_z": float(g["band60_dtw_rms_z"].median()),
                "median_axis_x_corr": float(g["axis_x_identity_corr"].median()),
                "median_axis_y_corr": float(g["axis_y_identity_corr"].median()),
                "median_axis_z_corr": float(g["axis_z_identity_corr"].median()),
                "median_axis_correction_m": float(g["band60_axis_median_abs_diff_m"].median()),
                "p90_axis_correction_m": float(g["band60_axis_p90_abs_diff_m"].median()),
            }
        )
    return pd.DataFrame(out).sort_values(["eval_tag", "median_band60_corr"], ascending=[True, False])


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    cross = summary[summary["eval_tag"] == "cross_5_13"].copy()
    if cross.empty:
        return
    cross = cross.sort_values("median_band60_corr", ascending=False)
    x = np.arange(len(cross))
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=180)
    width = 0.35
    ax.bar(x - width / 2, cross["median_identity_corr"], width, label="identity corr")
    ax.bar(x + width / 2, cross["median_band60_corr"], width, label="banded DTW corr")
    ax.set_xticks(x)
    ax.set_xticklabels(cross["candidate"], rotation=25, ha="right")
    ax.set_ylim(-0.2, 1.0)
    ax.set_ylabel("Median total-field correlation")
    ax.set_title("5.13 cross-day consistency for 4.14 anchor-map candidates")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_reference_examples(dist: np.ndarray, refs: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    keep = ["all_raw", "quality_good_exclude_bad", "forward_only", "top4_lopo_identity", "top6_lopo_identity"]
    fig, ax = plt.subplots(figsize=(12, 5), dpi=180)
    for name in keep:
        if name in refs:
            ax.plot(dist, dwd.z_valid(refs[name]["total_hp"]), lw=1.0, label=name)
    ax.set_xlabel("Along-track distance / m")
    ax.set_ylabel("Total HP robust z")
    ax.set_title("Candidate 4.14 reference maps")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_notes(summary: pd.DataFrame, sets: dict[str, list[str]], path: Path) -> None:
    cross = summary[summary["eval_tag"] == "cross_5_13"].sort_values("median_band60_corr", ascending=False)
    same = summary[summary["eval_tag"] == "same_4_14"].sort_values("median_band60_corr", ascending=False)
    lines = [
        "# Anchor Reference Selection Experiment",
        "",
        "Purpose: test whether some 4.14 passes pollute the averaged magnetic map. Candidate reference maps are built from fixed pass subsets, then evaluated against all 5.13 passes and all 4.14 passes.",
        "",
        "Method boundary:",
        "",
        "- This experiment does not use 5.13 truth to build the reference map. The candidate subsets are chosen from 4.14 same-day repeatability and direction metadata.",
        "- `identity_corr` means direct comparison at the current SPAN-derived distance axis.",
        "- `band60_dtw_corr` allows a bounded +/-60 m distance-axis correction and is only a map-quality diagnostic, not a deployable online localization result.",
        "",
        "Candidate pass sets:",
        "",
    ]
    for name, labels in sets.items():
        lines.append(f"- `{name}`: {', '.join(labels)}")
    lines += [
        "",
        "5.13 cross-day summary:",
        "",
        cross.to_markdown(index=False, floatfmt=".3f"),
        "",
        "4.14 same-day summary:",
        "",
        same.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Interpretation:",
        "",
        "- A useful anchor map should improve 5.13 cross-day metrics, not only 4.14 same-day metrics.",
        "- If `forward_only` is best, the current backward-axis transformation or backward distance axis is still suspect.",
        "- If `top*_lopo_identity` is best, repeatability-based map quality selection is a publishable preprocessing direction, but it still needs online localization validation.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dist4, passes4 = cma.load_passes("4_14")
    _, passes5 = cma.load_passes("5_13")
    by_seg = pass_by_label(passes4)
    sets = candidate_sets(passes4)
    refs = {name: build_ref(dist4, [by_seg[seg] for seg in labels]) for name, labels in sets.items()}

    eval_parts = []
    for name, labels in sets.items():
        selected = [by_seg[seg] for seg in labels]
        eval_parts.append(eval_candidate(name, refs[name], selected, passes5, "cross_5_13"))
        eval_parts.append(eval_candidate(name, refs[name], selected, passes4, "same_4_14"))
    rows = pd.concat(eval_parts, ignore_index=True)
    summary = summarize(rows)

    rows.to_csv(OUT_DIR / "anchor_reference_eval_by_segment.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "anchor_reference_eval_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "anchor_reference_eval_summary.json").write_text(
        json.dumps(
            {
                "candidate_sets": sets,
                "summary": summary.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_summary(summary, OUT_DIR / "anchor_reference_cross_day_summary.png")
    plot_reference_examples(dist4, refs, OUT_DIR / "anchor_reference_examples.png")
    write_notes(summary, sets, OUT_DIR / "anchor_reference_selection_notes.md")
    print(summary.round(3).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
