from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PROC = Path(r"C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\data_proc_new")
OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\axis_calibrated_experiment")
STEP_M = 0.5
WINDOWS_M = [20.0, 50.0, 100.0, 150.0]
QUERY_STRIDE_M = 5.0
PRIOR_RADII_M: list[float | None] = [None, 100.0, 50.0, 20.0]
BAD_4_14_REFERENCE_SEGMENTS = {"BMAW15230010L_1_seg02", "BMAW15230010L_1_seg03"}


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


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


def robust_z(x: np.ndarray, clip: float = 6.0) -> np.ndarray:
    med, scale = robust_center_scale(x)
    z = (np.asarray(x, dtype=float) - med) / scale
    return np.clip(z, -clip, clip)


def rolling_median(x: np.ndarray, points: int) -> np.ndarray:
    points = max(3, int(points) | 1)
    return (
        pd.Series(np.asarray(x, dtype=float))
        .rolling(points, center=True, min_periods=max(3, points // 4))
        .median()
        .to_numpy(float)
    )


def highpass_distance(x: np.ndarray, window_m: float = 35.0) -> np.ndarray:
    points = max(5, int(round(window_m / STEP_M)) | 1)
    return np.asarray(x, dtype=float) - rolling_median(x, points)


def load_map(date_tag: str) -> pd.DataFrame:
    return pd.read_csv(DATA_PROC / f"magmap_{date_tag}_0p5m.csv")


def load_segments(date_tag: str) -> pd.DataFrame:
    return pd.read_csv(DATA_PROC / f"magmap_{date_tag}_segments.csv")


def segment_columns(df: pd.DataFrame) -> list[str]:
    pattern = re.compile(r"^BMAW.*_seg\d+_mag_total$")
    return [c.replace("_mag_total", "") for c in df.columns if pattern.match(c)]


def date_of(tag: str) -> str:
    return "4_14" if tag == "4_14" else "5_13"


def map_body_5_13(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    direction: str,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Forward 5.13 was empirically stable: 4.14-X ~= 5.13-Z and
    # 4.14-Y ~= 5.13-Y. Backward is less constrained, so keep variants explicit.
    if variant == "fwd_z_y_x_back_z_y_minusx":
        return (z, y, x) if direction == "forward" else (z, y, -x)
    if variant == "fwd_z_y_x_back_minusy_minusz_minusx":
        return (z, y, x) if direction == "forward" else (-y, -z, -x)
    if variant == "all_z_y_x":
        return z, y, x
    raise ValueError(f"Unknown axis calibration variant: {variant}")


def segment_track_features(
    df: pd.DataFrame,
    seg: str,
    date_tag: str,
    direction: str,
    axis_variant: str = "fwd_z_y_x_back_z_y_minusx",
) -> dict[str, np.ndarray]:
    x = pd.to_numeric(df[f"{seg}_mag_x"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[f"{seg}_mag_y"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(df[f"{seg}_mag_z"], errors="coerce").to_numpy(float)
    total = pd.to_numeric(df[f"{seg}_mag_total"], errors="coerce").to_numpy(float)
    if date_tag == "5_13":
        x, y, z = map_body_5_13(x, y, z, direction, axis_variant)

    direction_sign = 1.0 if direction == "forward" else -1.0
    out: dict[str, np.ndarray] = {}
    for name, values in [("body_x_cal", x), ("body_y_cal", y), ("body_z_cal", z)]:
        med = float(np.nanmedian(values))
        out[name + "_anom"] = values - med

    out["track_x_anom"] = -direction_sign * out["body_x_cal_anom"]
    out["track_y_anom"] = out["body_y_cal_anom"]
    out["track_z_anom"] = direction_sign * out["body_z_cal_anom"]
    out["track_x_hp"] = highpass_distance(out["track_x_anom"], 35.0)
    out["track_y_hp"] = highpass_distance(out["track_y_anom"], 35.0)
    out["track_z_hp"] = highpass_distance(out["track_z_anom"], 35.0)
    out["total"] = total
    out["total_hp"] = highpass_distance(total, 45.0)
    return out


def segment_existing_track_features(df: pd.DataFrame, seg: str) -> dict[str, np.ndarray]:
    out = {
        "old_track_x": pd.to_numeric(df[f"{seg}_mag_x_track_anom"], errors="coerce").to_numpy(float),
        "old_track_y": pd.to_numeric(df[f"{seg}_mag_y_track_anom"], errors="coerce").to_numpy(float),
        "old_track_z": pd.to_numeric(df[f"{seg}_mag_z_track_anom"], errors="coerce").to_numpy(float),
        "total": pd.to_numeric(df[f"{seg}_mag_total"], errors="coerce").to_numpy(float),
    }
    out["old_track_x_hp"] = highpass_distance(out["old_track_x"], 35.0)
    out["old_track_y_hp"] = highpass_distance(out["old_track_y"], 35.0)
    out["old_track_z_hp"] = highpass_distance(out["old_track_z"], 35.0)
    out["total_hp"] = highpass_distance(out["total"], 45.0)
    return out


def mean_stack(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.array([])
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.vstack(arrays), axis=0)


def build_reference_features(
    axis_variant: str,
    reference_profile: str = "all",
) -> tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    df = load_map("4_14")
    segs = load_segments("4_14")
    dist = pd.to_numeric(df["distance_m"], errors="coerce").to_numpy(float)

    new_stacks: dict[str, list[np.ndarray]] = {k: [] for k in ["track_x_hp", "track_y_hp", "track_z_hp", "total_hp"]}
    old_stacks: dict[str, list[np.ndarray]] = {k: [] for k in ["old_track_x_hp", "old_track_y_hp", "old_track_z_hp", "total_hp"]}

    for _, row in segs.iterrows():
        seg = str(row["segment_label"])
        if reference_profile == "quality_good" and seg in BAD_4_14_REFERENCE_SEGMENTS:
            continue
        direction = str(row["direction"])
        new = segment_track_features(df, seg, "4_14", direction, axis_variant)
        old = segment_existing_track_features(df, seg)
        for key in new_stacks:
            new_stacks[key].append(new[key])
        for key in old_stacks:
            old_stacks[key].append(old[key])

    ref = {
        "axis_x_hp": mean_stack(new_stacks["track_x_hp"]),
        "axis_y_hp": mean_stack(new_stacks["track_y_hp"]),
        "axis_z_hp": mean_stack(new_stacks["track_z_hp"]),
        "axis_total_hp": mean_stack(new_stacks["total_hp"]),
        "old_x_hp": mean_stack(old_stacks["old_track_x_hp"]),
        "old_y_hp": mean_stack(old_stacks["old_track_y_hp"]),
        "old_z_hp": mean_stack(old_stacks["old_track_z_hp"]),
        "total_hp": mean_stack(old_stacks["total_hp"]),
        "total_raw": pd.to_numeric(df["map_mag_total_mean_nT"], errors="coerce").to_numpy(float),
    }
    ref["total_raw_hp"] = highpass_distance(ref["total_raw"], 45.0)
    return dist, ref, segs


@dataclass
class QueryFeature:
    segment: str
    direction: str
    distance: np.ndarray
    features: dict[str, np.ndarray]


def build_query_features(axis_variant: str) -> list[QueryFeature]:
    df = load_map("5_13")
    segs = load_segments("5_13")
    dist = pd.to_numeric(df["distance_m"], errors="coerce").to_numpy(float)
    out: list[QueryFeature] = []
    for _, row in segs.iterrows():
        seg = str(row["segment_label"])
        direction = str(row["direction"])
        new = segment_track_features(df, seg, "5_13", direction, axis_variant)
        old = segment_existing_track_features(df, seg)
        features = {
            "axis_x_hp": new["track_x_hp"],
            "axis_y_hp": new["track_y_hp"],
            "axis_z_hp": new["track_z_hp"],
            "axis_total_hp": new["total_hp"],
            "old_x_hp": old["old_track_x_hp"],
            "old_y_hp": old["old_track_y_hp"],
            "old_z_hp": old["old_track_z_hp"],
            "total_hp": old["total_hp"],
            "total_raw": old["total"],
            "total_raw_hp": old["total_hp"],
        }
        out.append(QueryFeature(seg, direction, dist, features))
    return out


FEATURE_SETS: dict[str, list[str]] = {
    "total_raw_hp_ncc": ["total_raw_hp"],
    "old_track_xy_hp_ncc": ["old_x_hp", "old_y_hp"],
    "axiscal_xy_hp_ncc": ["axis_x_hp", "axis_y_hp"],
    "axiscal_xy_total_hp_ncc": ["axis_x_hp", "axis_y_hp", "axis_total_hp"],
    "axiscal_xyz_hp_ncc": ["axis_x_hp", "axis_y_hp", "axis_z_hp"],
    "axiscal_xy_hp_msd": ["axis_x_hp", "axis_y_hp"],
}


def score_pair(q_feats: list[np.ndarray], r_feats: list[np.ndarray], mode: str) -> float:
    scores = []
    for q, r in zip(q_feats, r_feats):
        mask = np.isfinite(q) & np.isfinite(r)
        if mask.sum() < max(10, int(0.8 * len(q))):
            return math.nan
        qz = robust_z(q[mask])
        rz = robust_z(r[mask])
        if mode == "ncc":
            scores.append(float(np.nanmean(qz * rz)))
        elif mode == "msd":
            scores.append(float(-np.nanmean((qz - rz) ** 2)))
        else:
            raise ValueError(mode)
    return float(np.nanmean(scores))


def z_window(v: np.ndarray) -> np.ndarray | None:
    if not np.isfinite(v).all():
        return None
    z = robust_z(v)
    if not np.isfinite(z).all():
        return None
    return z


def reference_window_matrices(
    ref_dist: np.ndarray,
    ref_features: dict[str, np.ndarray],
    feature_names: list[str],
    n: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    starts = []
    rows_by_feat: list[list[np.ndarray]] = [[] for _ in feature_names]
    for i in range(0, len(ref_dist) - n + 1):
        feat_windows = []
        ok = True
        for name in feature_names:
            zw = z_window(ref_features[name][i : i + n])
            if zw is None:
                ok = False
                break
            feat_windows.append(zw)
        if not ok:
            continue
        starts.append(float(ref_dist[i]))
        for j, zw in enumerate(feat_windows):
            rows_by_feat[j].append(zw)
    if not starts:
        return np.array([], dtype=float), []
    matrices = [np.vstack(rows).astype(np.float32) for rows in rows_by_feat]
    return np.asarray(starts, dtype=float), matrices


def best_match(
    query_window: list[np.ndarray],
    ref_starts: np.ndarray,
    ref_matrices: list[np.ndarray],
    mode: str,
    true_start: float,
    prior_radius_m: float | None,
) -> dict[str, float]:
    if len(ref_matrices) == 0 or len(ref_starts) == 0:
        return {"pred_start_m": math.nan, "best_score": math.nan, "second_score": math.nan}
    qz_windows = []
    for w in query_window:
        zw = z_window(w)
        if zw is None:
            return {"pred_start_m": math.nan, "best_score": math.nan, "second_score": math.nan}
        qz_windows.append(zw.astype(np.float32))

    scores = np.zeros(len(ref_starts), dtype=np.float32)
    for qz, mat in zip(qz_windows, ref_matrices):
        if mode == "ncc":
            scores += mat @ qz / float(len(qz))
        elif mode == "msd":
            scores += -np.mean((mat - qz[None, :]) ** 2, axis=1)
        else:
            raise ValueError(mode)
    scores /= float(len(ref_matrices))
    scores = scores.astype(float)

    cand_mask = np.ones(len(ref_starts), dtype=bool)
    if prior_radius_m is not None:
        cand_mask &= np.abs(ref_starts - true_start) <= prior_radius_m
    scores[~cand_mask] = np.nan
    if not np.isfinite(scores).any():
        return {"pred_start_m": math.nan, "best_score": math.nan, "second_score": math.nan}
    best_idx = int(np.nanargmax(scores))
    best_start = float(ref_starts[best_idx])
    distinct = cand_mask & (np.abs(ref_starts - best_start) >= 10.0)
    second_score = math.nan
    if distinct.any() and np.isfinite(scores[distinct]).any():
        second_score = float(np.nanmax(scores[distinct]))
    return {
        "pred_start_m": best_start,
        "best_score": float(scores[best_idx]),
        "second_score": second_score,
    }


def validate_variant(
    axis_variant: str,
    reference_profile: str = "all",
    feature_sets: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_sets = FEATURE_SETS if feature_sets is None else feature_sets
    ref_dist, ref_features, _ = build_reference_features(axis_variant, reference_profile=reference_profile)
    queries = build_query_features(axis_variant)
    rows = []
    stride_n = max(1, int(round(QUERY_STRIDE_M / STEP_M)))

    for window_m in WINDOWS_M:
        n = int(round(window_m / STEP_M)) + 1
        ref_cache: dict[str, tuple[np.ndarray, list[np.ndarray]]] = {}
        for method, feats in feature_sets.items():
            ref_cache[method] = reference_window_matrices(ref_dist, ref_features, feats, n)

        for q in queries:
            for method, feats in feature_sets.items():
                mode = "msd" if method.endswith("_msd") else "ncc"
                ref_starts, ref_wins = ref_cache[method]
                q_arrays = [q.features[name] for name in feats]
                finite = np.isfinite(np.column_stack(q_arrays)).all(axis=1)
                if finite.sum() < n:
                    continue
                for i in range(0, len(q.distance) - n + 1, stride_n):
                    d_win = q.distance[i : i + n]
                    q_win = [arr[i : i + n] for arr in q_arrays]
                    if np.nanmax(d_win) - np.nanmin(d_win) < window_m - STEP_M:
                        continue
                    if not all(np.isfinite(w).sum() >= max(10, int(0.8 * n)) for w in q_win):
                        continue
                    true_start = float(d_win[0])
                    for prior in PRIOR_RADII_M:
                        res = best_match(q_win, ref_starts, ref_wins, mode, true_start, prior)
                        pred = res["pred_start_m"]
                        err = pred - true_start if np.isfinite(pred) else math.nan
                        rows.append(
                            {
                                "axis_variant": axis_variant,
                                "reference_profile": reference_profile,
                                "method": method,
                                "mode": mode,
                                "prior_radius_m": "global" if prior is None else f"plusminus_{prior:g}",
                                "query_segment": q.segment,
                                "query_direction": q.direction,
                                "window_m": window_m,
                                "true_start_m": true_start,
                                "true_end_m": float(d_win[-1]),
                                "pred_start_m": pred,
                                "error_m": err,
                                "abs_error_m": abs(err) if np.isfinite(err) else math.nan,
                                "best_score": res["best_score"],
                                "second_score": res["second_score"],
                                "score_gap": res["best_score"] - res["second_score"]
                                if np.isfinite(res["second_score"])
                                else math.nan,
                            }
                        )
    results = pd.DataFrame(rows)
    summary = summarize(results)
    return results, summary


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    groups = ["axis_variant", "reference_profile", "method", "prior_radius_m", "window_m"]
    rows = []
    for key, g in results.groupby(groups, dropna=False):
        valid = g.dropna(subset=["abs_error_m"])
        if valid.empty:
            continue
        row = dict(zip(groups, key))
        row.update(
            {
                "query_count": int(len(valid)),
                "median_abs_error_m": float(valid["abs_error_m"].median()),
                "mean_abs_error_m": float(valid["abs_error_m"].mean()),
                "p75_abs_error_m": float(valid["abs_error_m"].quantile(0.75)),
                "p90_abs_error_m": float(valid["abs_error_m"].quantile(0.90)),
                "rmse_m": float(np.sqrt(np.mean(np.square(valid["error_m"].to_numpy(float))))),
                "median_best_score": float(valid["best_score"].median()),
                "median_score_gap": float(valid["score_gap"].median()),
                "forward_median_abs_error_m": float(
                    valid.loc[valid["query_direction"] == "forward", "abs_error_m"].median()
                ),
                "backward_median_abs_error_m": float(
                    valid.loc[valid["query_direction"] == "backward", "abs_error_m"].median()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["prior_radius_m", "window_m", "median_abs_error_m"])


def plot_summary(summary: pd.DataFrame, axis_variant: str, reference_profile: str) -> list[Path]:
    paths: list[Path] = []
    if summary.empty:
        return paths
    for prior in ["global", "plusminus_100", "plusminus_50", "plusminus_20"]:
        sub = summary[
            (summary["axis_variant"] == axis_variant)
            & (summary["reference_profile"] == reference_profile)
            & (summary["prior_radius_m"] == prior)
        ].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
        for method, g in sub.groupby("method"):
            g = g.sort_values("window_m")
            ax.plot(g["window_m"], g["median_abs_error_m"], marker="o", label=method)
        ax.set_title(
            f"4.14 -> 5.13 axis calibration matching, prior={prior}, "
            f"variant={axis_variant}, ref={reference_profile}"
        )
        ax.set_xlabel("Window length / m")
        ax.set_ylabel("Median absolute position error / m")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = OUT_DIR / f"summary_{reference_profile}_{axis_variant}_{prior}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def write_notes(best_summary: pd.DataFrame, paths: list[Path]) -> None:
    note = OUT_DIR / "axis_calibrated_experiment_notes.md"
    best_rows = best_summary.head(20).copy()
    lines = [
        "# Axis-Calibrated Cross-Day Matching Experiment",
        "",
        "This experiment keeps the existing SPAN-time aligned 0.5 m grid fixed and only changes the magnetic feature coordinates.",
        "",
        "Main tested idea: remap 5.13 magnetic axes before direction rotation, then match robust high-pass X/Y signatures.",
        "",
        "Important caution: prior-limited results use the true position only to emulate an external weak prior/search window. They are not standalone localization results.",
        "",
        "## Best Summary Rows",
        "",
        best_rows.to_markdown(index=False),
        "",
        "## Figures",
        "",
    ]
    for p in paths:
        lines.append(f"- `{p}`")
    note.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Axis-calibrated magnetic map matching experiment")
    parser.add_argument("--reference-profile", choices=["all", "quality_good"], default="all")
    parser.add_argument(
        "--axis-variants",
        nargs="*",
        default=["fwd_z_y_x_back_z_y_minusx", "fwd_z_y_x_back_minusy_minusz_minusx"],
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(FEATURE_SETS.keys()),
        help="Subset of methods to run. Default: all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = args.axis_variants
    selected_features = {m: FEATURE_SETS[m] for m in args.methods if m in FEATURE_SETS}
    if not selected_features:
        raise ValueError(f"No valid methods selected from: {args.methods}")
    all_results = []
    all_summaries = []
    fig_paths: list[Path] = []
    for variant in variants:
        results, summary = validate_variant(
            variant,
            reference_profile=args.reference_profile,
            feature_sets=selected_features,
        )
        prefix = f"{args.reference_profile}_{variant}"
        results.to_csv(OUT_DIR / f"matching_results_{prefix}.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(OUT_DIR / f"matching_summary_{prefix}.csv", index=False, encoding="utf-8-sig")
        all_results.append(results)
        all_summaries.append(summary)
        fig_paths.extend(plot_summary(summary, variant, args.reference_profile))
    results_all = pd.concat(all_results, ignore_index=True)
    summary_all = pd.concat(all_summaries, ignore_index=True)
    all_prefix = f"{args.reference_profile}_matching"
    results_all.to_csv(OUT_DIR / f"{all_prefix}_results_all_axis_variants.csv", index=False, encoding="utf-8-sig")
    summary_all.to_csv(OUT_DIR / f"{all_prefix}_summary_all_axis_variants.csv", index=False, encoding="utf-8-sig")
    summary_json = json.loads(summary_all.head(100).to_json(orient="records", force_ascii=False))
    (OUT_DIR / f"{all_prefix}_summary_head.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_notes(summary_all.sort_values(["prior_radius_m", "window_m", "median_abs_error_m"]), fig_paths)

    print("Best rows by prior/window/method:")
    cols = [
        "axis_variant",
        "reference_profile",
        "method",
        "prior_radius_m",
        "window_m",
        "query_count",
        "median_abs_error_m",
        "p75_abs_error_m",
        "rmse_m",
        "forward_median_abs_error_m",
        "backward_median_abs_error_m",
        "median_score_gap",
    ]
    print(summary_all.sort_values(["prior_radius_m", "window_m", "median_abs_error_m"])[cols].head(40).round(3).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
