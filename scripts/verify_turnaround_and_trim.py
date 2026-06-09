from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import anchor_reference_hmm_experiment as arh
import axis_calibrated_hmm_experiment as hmm


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\turnaround_and_trim_diagnostic")
AXIS_VARIANT = "fwd_z_y_x_back_z_y_minusx"
SAMPLE_PERIOD = "4s"


def find_file(filename: str) -> Path:
    root = Path.home() / "Desktop" / "磁导航" / "数据" / "codex_railway_magnav" / "data_proc_new"
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(filename)
    return matches[0]


def parse_vmax(method: str) -> float:
    m = re.search(r"vmax([0-9.]+)", method)
    if not m:
        raise ValueError(method)
    return float(m.group(1))


def robust_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 20:
        return math.nan
    aa = a[mask] - np.nanmedian(a[mask])
    bb = b[mask] - np.nanmedian(b[mask])
    sa = np.nanstd(aa)
    sb = np.nanstd(bb)
    if sa < 1e-9 or sb < 1e-9:
        return math.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def load_direction_maps(path: Path, date_label: str, bin_m: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = ["mag_x", "mag_y", "mag_z", "mag_total", "s_abs_m", "segment_label", "direction", "yaw"]
    chunks = []
    yaw_rows = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=350_000):
        chunk = chunk.dropna(subset=["mag_x", "mag_y", "mag_z", "s_abs_m", "segment_label", "direction"])
        if chunk.empty:
            continue
        chunk["s_bin_m"] = (chunk["s_abs_m"] / bin_m).round() * bin_m
        for col in ["mag_x", "mag_y", "mag_z", "mag_total"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        grouped = (
            chunk.groupby(["segment_label", "direction", "s_bin_m"], observed=True)
            .agg(
                mag_x=("mag_x", "mean"),
                mag_y=("mag_y", "mean"),
                mag_z=("mag_z", "mean"),
                mag_total=("mag_total", "mean"),
                count=("mag_x", "size"),
            )
            .reset_index()
        )
        chunks.append(grouped)
        if "yaw" in chunk.columns:
            yr = (
                chunk.dropna(subset=["yaw"])
                .groupby(["segment_label", "direction"], observed=True)
                .agg(yaw_mean=("yaw", "mean"), yaw_median=("yaw", "median"), n=("yaw", "size"))
                .reset_index()
            )
            yaw_rows.append(yr)
    if not chunks:
        return pd.DataFrame(), pd.DataFrame()
    binned = pd.concat(chunks, ignore_index=True)
    # Merge chunk means by weighted average.
    for col in ["mag_x", "mag_y", "mag_z", "mag_total"]:
        binned[col + "_sum"] = binned[col] * binned["count"]
    binned = (
        binned.groupby(["segment_label", "direction", "s_bin_m"], observed=True)
        .agg(
            mag_x_sum=("mag_x_sum", "sum"),
            mag_y_sum=("mag_y_sum", "sum"),
            mag_z_sum=("mag_z_sum", "sum"),
            mag_total_sum=("mag_total_sum", "sum"),
            count=("count", "sum"),
        )
        .reset_index()
    )
    for col in ["mag_x", "mag_y", "mag_z", "mag_total"]:
        binned[col] = binned[col + "_sum"] / binned["count"]
    binned["date"] = date_label
    yaw = pd.concat(yaw_rows, ignore_index=True) if yaw_rows else pd.DataFrame()
    if not yaw.empty:
        yaw = (
            yaw.groupby(["segment_label", "direction"], observed=True)
            .apply(lambda g: pd.Series({"yaw_mean": np.average(g["yaw_mean"], weights=g["n"]), "yaw_median": np.average(g["yaw_median"], weights=g["n"]), "n": int(g["n"].sum())}))
            .reset_index()
        )
        yaw["date"] = date_label
    return binned[["date", "segment_label", "direction", "s_bin_m", "mag_x", "mag_y", "mag_z", "mag_total", "count"]], yaw


def direction_axis_correlation(binned: pd.DataFrame) -> pd.DataFrame:
    if binned.empty:
        return pd.DataFrame()
    work = binned.copy()
    for col in ["mag_x", "mag_y", "mag_z", "mag_total"]:
        work[col + "_anom"] = work[col] - work.groupby("segment_label")[col].transform("median")
    maps = (
        work.groupby(["date", "direction", "s_bin_m"], observed=True)
        .agg(
            x=("mag_x_anom", "mean"),
            y=("mag_y_anom", "mean"),
            z=("mag_z_anom", "mean"),
            total=("mag_total_anom", "mean"),
            pass_bins=("segment_label", "nunique"),
        )
        .reset_index()
    )
    rows = []
    for date, g in maps.groupby("date"):
        if {"forward", "backward"} - set(g["direction"]):
            continue
        f = g[g["direction"] == "forward"].set_index("s_bin_m")
        b = g[g["direction"] == "backward"].set_index("s_bin_m")
        common = f.index.intersection(b.index)
        common = common[(common >= max(f.index.min(), b.index.min())) & (common <= min(f.index.max(), b.index.max()))]
        for col in ["x", "y", "z", "total"]:
            corr_same = robust_corr(f.loc[common, col].to_numpy(float), b.loc[common, col].to_numpy(float))
            rows.append(
                {
                    "date": date,
                    "axis": col,
                    "corr_forward_vs_backward_same_sign": corr_same,
                    "corr_forward_vs_backward_flipped": -corr_same if np.isfinite(corr_same) else math.nan,
                    "common_bins": int(len(common)),
                    "interpretation": "same_body_orientation_likely" if corr_same > 0.2 else ("turnaround_or_axis_flip_likely" if corr_same < -0.2 else "unclear"),
                }
            )
    return pd.DataFrame(rows)


def pairwise_direction_correlation(binned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if binned.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = binned.copy()
    for col in ["mag_x", "mag_y", "mag_z", "mag_total"]:
        work[col + "_anom"] = work[col] - work.groupby("segment_label")[col].transform("median")
    rows = []
    for date, dg in work.groupby("date"):
        fwds = sorted(dg.loc[dg["direction"] == "forward", "segment_label"].unique())
        backs = sorted(dg.loc[dg["direction"] == "backward", "segment_label"].unique())
        for fseg in fwds:
            f = dg[dg["segment_label"] == fseg].set_index("s_bin_m")
            for bseg in backs:
                b = dg[dg["segment_label"] == bseg].set_index("s_bin_m")
                common = f.index.intersection(b.index)
                if len(common) < 120:
                    continue
                row = {"date": date, "forward_segment": fseg, "backward_segment": bseg, "common_bins": int(len(common))}
                for axis, col in [("x", "mag_x_anom"), ("y", "mag_y_anom"), ("z", "mag_z_anom"), ("total", "mag_total_anom")]:
                    row[f"corr_{axis}"] = robust_corr(f.loc[common, col].to_numpy(float), b.loc[common, col].to_numpy(float))
                rows.append(row)
    pairwise = pd.DataFrame(rows)
    if pairwise.empty:
        return pairwise, pd.DataFrame()
    summary_rows = []
    for date, g in pairwise.groupby("date"):
        item = {"date": date, "pair_count": int(len(g))}
        for axis in ["x", "y", "z", "total"]:
            vals = g[f"corr_{axis}"].dropna()
            item[f"{axis}_median_corr"] = float(vals.median()) if len(vals) else math.nan
            item[f"{axis}_positive_pair_count"] = int((vals > 0.2).sum())
            item[f"{axis}_negative_pair_count"] = int((vals < -0.2).sum())
        summary_rows.append(item)
    return pairwise, pd.DataFrame(summary_rows)


def trim_stop_index(q: hmm.QuerySegment, threshold_m: float = 20.0) -> tuple[int, dict[str, float]]:
    s = np.asarray(q.truth_s, dtype=float)
    if q.direction == "backward":
        idx = int(np.nanargmin(s))
        reverse_m = float(s[-1] - s[idx])
        should_trim = reverse_m > threshold_m and idx < len(s) - 10
    else:
        idx = int(np.nanargmax(s))
        reverse_m = float(s[idx] - s[-1])
        should_trim = reverse_m > threshold_m and idx < len(s) - 10
    stop = idx + 1 if should_trim else len(s)
    duration_tail_s = float((pd.Timestamp(q.time[-1]) - pd.Timestamp(q.time[idx])).total_seconds()) if should_trim else 0.0
    return stop, {
        "trim_applied": float(should_trim),
        "trim_stop_index": int(stop),
        "reverse_tail_distance_m": reverse_m if should_trim else 0.0,
        "reverse_tail_duration_s": duration_tail_s,
        "s_start_m": float(s[0]),
        "s_end_m": float(s[-1]),
        "s_endpoint_m": float(s[idx]),
    }


def eval_pred(pred: np.ndarray, truth: np.ndarray, stop: int | None = None) -> dict[str, float]:
    if stop is not None:
        pred = pred[:stop]
        truth = truth[:stop]
    warmup = min(20, max(0, len(pred) // 10))
    mask = np.isfinite(pred) & np.isfinite(truth)
    mask[:warmup] = False
    err = pred[mask] - truth[mask]
    return {
        "sample_count": int(len(err)),
        "median_abs_error_m": float(np.median(np.abs(err))),
        "mean_abs_error_m": float(np.mean(np.abs(err))),
        "rmse_m": float(np.sqrt(np.mean(err * err))),
        "p90_abs_error_m": float(np.percentile(np.abs(err), 90)),
        "final_abs_error_m": float(abs(err[-1])),
        "max_abs_error_m": float(np.max(np.abs(err))),
        "within_25m_rate": float(np.mean(np.abs(err) <= 25.0)),
        "within_50m_rate": float(np.mean(np.abs(err) <= 50.0)),
    }


def run_selected_candidate(q: hmm.QuerySegment, total_ref: dict[str, np.ndarray], axis_ref: dict[str, np.ndarray], selected: str) -> np.ndarray:
    if selected == "AxisMidGate":
        pred, _ = hmm.viterbi_track(
            q,
            axis_ref,
            ["axis_x_hp_z", "axis_y_hp_z", "axis_total_hp_z"],
            {"axis_x_hp_z": 0.8, "axis_y_hp_z": 0.8, "axis_total_hp_z": 1.0},
            sigma=1.35,
            vmax_mps=1.4,
            robust=True,
            info_gate=True,
            gate_min_scale=0.30,
            gate_offset=0.02,
            gate_span=0.24,
            start_prior="uniform",
        )
        return pred
    vmax = parse_vmax(selected)
    pred, _ = hmm.viterbi_track(
        q,
        total_ref,
        ["total_raw_hp_z"],
        {"total_raw_hp_z": 1.0},
        sigma=1.2,
        vmax_mps=vmax,
        robust=True,
        info_gate=False,
        start_prior="uniform",
    )
    return pred


def tail_gpgga_quality(path_5_13: Path, segment_label: str) -> dict[str, float]:
    usecols = ["time", "segment_label", "s_abs_m", "fix_quality", "satellites", "hdop"]
    frames = []
    for chunk in pd.read_csv(path_5_13, usecols=usecols, chunksize=350_000):
        part = chunk[chunk["segment_label"] == segment_label].copy()
        if not part.empty:
            frames.append(part)
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "s_abs_m"]).sort_values("time")
    idx = int(df["s_abs_m"].idxmin())
    tail = df.loc[df.index >= idx]
    return {
        "tail_raw_sample_count": int(len(tail)),
        "tail_s_min_m": float(df.loc[idx, "s_abs_m"]),
        "tail_s_end_m": float(df["s_abs_m"].iloc[-1]),
        "tail_reverse_distance_m": float(df["s_abs_m"].iloc[-1] - df.loc[idx, "s_abs_m"]),
        "tail_fix_quality_median": float(pd.to_numeric(tail["fix_quality"], errors="coerce").median()),
        "tail_satellites_median": float(pd.to_numeric(tail["satellites"], errors="coerce").median()),
        "tail_hdop_median": float(pd.to_numeric(tail["hdop"], errors="coerce").median()),
    }


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path_4 = find_file("magmap_4_14_aligned_samples.csv")
    path_5 = find_file("magmap_5_13_aligned_samples.csv")

    b4, yaw4 = load_direction_maps(path_4, "4.14")
    b5, yaw5 = load_direction_maps(path_5, "5.13")
    binned = pd.concat([b4, b5], ignore_index=True)
    yaw = pd.concat([yaw4, yaw5], ignore_index=True)
    corr = direction_axis_correlation(binned)
    pairwise_corr, pairwise_summary = pairwise_direction_correlation(binned)
    corr.to_csv(OUT_DIR / "direction_axis_correlation.csv", index=False, encoding="utf-8-sig")
    pairwise_corr.to_csv(OUT_DIR / "pairwise_forward_backward_axis_correlation.csv", index=False, encoding="utf-8-sig")
    pairwise_summary.to_csv(OUT_DIR / "pairwise_forward_backward_axis_summary.csv", index=False, encoding="utf-8-sig")
    yaw.to_csv(OUT_DIR / "segment_yaw_summary.csv", index=False, encoding="utf-8-sig")

    refs, _ = arh.build_candidate_refs()
    total_ref = refs["forward_only"]
    axis_ref = hmm.build_reference(AXIS_VARIANT, "all")
    decisions = pd.read_csv("progress_margin_selector_experiment/progress_margin_selector_decisions.csv")
    queries = hmm.read_query_segments(AXIS_VARIANT, SAMPLE_PERIOD)

    trim_rows = []
    metric_rows = []
    for q in queries:
        selected = str(decisions.loc[decisions["segment_label"] == q.label, "selected_candidate"].iloc[0])
        pred = run_selected_candidate(q, total_ref, axis_ref, selected)
        stop, trim_meta = trim_stop_index(q)
        trim_rows.append({"segment_label": q.label, "direction": q.direction, "selected_candidate": selected, **trim_meta})
        metric_rows.append({"metric_set": "full_segment", "segment_label": q.label, "direction": q.direction, "selected_candidate": selected, **eval_pred(pred, q.truth_s)})
        metric_rows.append({"metric_set": "trim_reversal_tail", "segment_label": q.label, "direction": q.direction, "selected_candidate": selected, **eval_pred(pred, q.truth_s, stop=stop)})

    trim_df = pd.DataFrame(trim_rows)
    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby("metric_set")
        .agg(
            segment_count=("segment_label", "size"),
            median_abs_error_m=("median_abs_error_m", "median"),
            mean_abs_error_m=("mean_abs_error_m", "mean"),
            rmse_m=("rmse_m", "mean"),
            median_final_error_m=("final_abs_error_m", "median"),
            mean_final_error_m=("final_abs_error_m", "mean"),
            max_final_error_m=("final_abs_error_m", "max"),
            mean_within_25m_rate=("within_25m_rate", "mean"),
            mean_within_50m_rate=("within_50m_rate", "mean"),
        )
        .reset_index()
    )
    subset_rows = []
    for subset_name, subset_df in [
        ("all", metrics),
        ("exclude_1_seg03", metrics[~metrics["segment_label"].str.contains("1_seg03")]),
    ]:
        for metric_set, g in subset_df.groupby("metric_set"):
            subset_rows.append(
                {
                    "subset": subset_name,
                    "metric_set": metric_set,
                    "segment_count": int(len(g)),
                    "median_abs_error_m": float(g["median_abs_error_m"].median()),
                    "mean_abs_error_m": float(g["mean_abs_error_m"].mean()),
                    "rmse_m": float(g["rmse_m"].mean()),
                    "median_final_error_m": float(g["final_abs_error_m"].median()),
                    "mean_final_error_m": float(g["final_abs_error_m"].mean()),
                    "max_final_error_m": float(g["final_abs_error_m"].max()),
                    "mean_within_25m_rate": float(g["within_25m_rate"].mean()),
                    "mean_within_50m_rate": float(g["within_50m_rate"].mean()),
                }
            )
    subset_summary = pd.DataFrame(subset_rows)
    quality_9 = tail_gpgga_quality(path_5, "BMAW15230010L_9_seg01")
    trim_df.to_csv(OUT_DIR / "reversal_tail_trim_flags.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT_DIR / "progress_margin_selector_trimmed_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "progress_margin_selector_trimmed_summary.csv", index=False, encoding="utf-8-sig")
    subset_summary.to_csv(OUT_DIR / "progress_margin_selector_trimmed_subset_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "turnaround_and_trim_summary.json").write_text(
        json.dumps(
            {
                "direction_axis_correlation": corr.to_dict(orient="records"),
                "pairwise_forward_backward_axis_summary": pairwise_summary.to_dict(orient="records"),
                "trim_flags": trim_df.to_dict(orient="records"),
                "trimmed_summary": summary.to_dict(orient="records"),
                "trimmed_subset_summary": subset_summary.to_dict(orient="records"),
                "gpgga_tail_quality_9_seg01": quality_9,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nDirection axis correlation:")
    print(corr.round(3).to_string(index=False))
    print("\nPairwise direction axis summary:")
    print(pairwise_summary.round(3).to_string(index=False))
    print("\nTrim flags:")
    print(trim_df.round(3).to_string(index=False))
    print("\nTrim summary:")
    print(summary.round(3).to_string(index=False))
    print("\nTrim subset summary:")
    print(subset_summary.round(3).to_string(index=False))
    print("\n9_seg01 GPGGA tail quality:")
    print(json.dumps(quality_9, ensure_ascii=False, indent=2))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
