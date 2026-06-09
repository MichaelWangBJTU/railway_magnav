from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import axis_calibrated_hmm_experiment as hmm


ROOT = Path(r"C:\Users\m1352\Documents\railway_magnav")
IN_TRAJ = ROOT / "axis_calibrated_hmm_speed_prior" / "axis_calibrated_hmm_trajectories.csv"
OUT_DIR = ROOT / "reliability_learning_experiment"

METHODS = [
    "Baseline_TotalHP_Viterbi",
    "SpeedPrior_TotalHP_Viterbi",
    "TotalHP_InfoGate_Viterbi",
    "AxisCal_XY_TotalHP_MidGate_Viterbi",
]
GOOD_ERROR_M = 25.0


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def build_candidate_table() -> pd.DataFrame:
    df = pd.read_csv(IN_TRAJ)
    df = df[df["method"].isin(METHODS)].copy()
    df["abs_error_m"] = df["error_m"].abs()
    df["good"] = (df["abs_error_m"] <= GOOD_ERROR_M).astype(int)
    key = ["segment_label", "time"]
    piv = df.pivot_table(index=key, columns="method", values="pred_s_m", aggfunc="first")
    piv = piv.dropna(subset=METHODS, how="any").reset_index()
    truth = df.groupby(key)["truth_s_m"].first().reset_index()
    direction = df.groupby(key)["direction"].first().reset_index()
    base = piv.merge(truth, on=key).merge(direction, on=key)
    pred_values = base[METHODS].to_numpy(float)
    base["pred_median"] = np.median(pred_values, axis=1)
    base["pred_range"] = np.max(pred_values, axis=1) - np.min(pred_values, axis=1)
    for m in METHODS:
        others = [x for x in METHODS if x != m]
        base[f"{m}_to_median"] = (base[m] - base["pred_median"]).abs()
        base[f"{m}_nearest_diff"] = np.min(np.abs(base[others].to_numpy(float) - base[[m]].to_numpy(float)), axis=1)

    rows = []
    for _, row in base.iterrows():
        seg_df = df[(df["segment_label"] == row["segment_label"]) & (df["time"] == row["time"])].set_index("method")
        for m in METHODS:
            err = float(seg_df.loc[m, "abs_error_m"])
            rec = {
                "segment_label": row["segment_label"],
                "time": row["time"],
                "direction": row["direction"],
                "method": m,
                "truth_s_m": float(row["truth_s_m"]),
                "pred_s_m": float(row[m]),
                "abs_error_m": err,
                "good": int(err <= GOOD_ERROR_M),
                "pred_range": float(row["pred_range"]),
                "to_median": float(row[f"{m}_to_median"]),
                "nearest_diff": float(row[f"{m}_nearest_diff"]),
                "method_baseline": int(m == "Baseline_TotalHP_Viterbi"),
                "method_speed": int(m == "SpeedPrior_TotalHP_Viterbi"),
                "method_total_gate": int(m == "TotalHP_InfoGate_Viterbi"),
                "method_axis_gate": int(m == "AxisCal_XY_TotalHP_MidGate_Viterbi"),
                "direction_forward": int(row["direction"] == "forward"),
            }
            rows.append(rec)
    out = pd.DataFrame(rows)
    out["sample_id"] = out["segment_label"].astype(str) + "|" + out["time"].astype(str)
    out = add_measurement_confidence_features(out)
    return out


def method_measurement_config(method: str) -> tuple[list[str], dict[str, float], float]:
    if method in {"Baseline_TotalHP_Viterbi", "SpeedPrior_TotalHP_Viterbi", "TotalHP_InfoGate_Viterbi"}:
        return ["total_raw_hp_z"], {"total_raw_hp_z": 1.0}, 1.2
    if method == "AxisCal_XY_TotalHP_MidGate_Viterbi":
        return ["axis_x_hp_z", "axis_y_hp_z", "axis_total_hp_z"], {
            "axis_x_hp_z": 0.8,
            "axis_y_hp_z": 0.8,
            "axis_total_hp_z": 1.0,
        }, 1.35
    raise ValueError(method)


def measurement_row_margin(row: np.ndarray, dist: np.ndarray, exclude_m: float = 30.0) -> tuple[int, float]:
    best_i = int(np.nanargmax(row))
    far = np.abs(dist - dist[best_i]) >= exclude_m
    if far.any() and np.isfinite(row[far]).any():
        second = float(np.nanmax(row[far]))
        return best_i, float(row[best_i] - second)
    return best_i, np.nan


def add_measurement_confidence_features(cand: pd.DataFrame) -> pd.DataFrame:
    ref = hmm.build_reference("fwd_z_y_x_back_z_y_minusx", "all")
    dist = ref["distance_m"]
    queries = {q.label: q for q in hmm.read_query_segments("fwd_z_y_x_back_z_y_minusx", "4s")}
    frames = []
    for seg, part in cand.groupby("segment_label", sort=False):
        q = queries.get(seg)
        if q is None:
            frames.append(part)
            continue
        q_times = pd.to_datetime(q.time)
        time_to_idx = {str(pd.Timestamp(t)): i for i, t in enumerate(q_times)}
        part = part.copy()
        part["ll_at_pred"] = np.nan
        part["meas_best_s_m"] = np.nan
        part["pred_to_meas_best_m"] = np.nan
        part["meas_margin_30m"] = np.nan
        for method, mpart in part.groupby("method"):
            features, weights, sigma = method_measurement_config(method)
            ll = hmm.measurement_loglikelihood(q, ref, features, weights, sigma=sigma, robust=True)
            best_idx = np.zeros(len(q.time), dtype=int)
            margins = np.full(len(q.time), np.nan)
            for k in range(len(q.time)):
                best_idx[k], margins[k] = measurement_row_margin(ll[k], dist)
            for idx, row in mpart.iterrows():
                k = time_to_idx.get(str(pd.Timestamp(row["time"])))
                if k is None:
                    continue
                pred_i = int(np.argmin(np.abs(dist - float(row["pred_s_m"]))))
                part.loc[idx, "ll_at_pred"] = float(ll[k, pred_i])
                part.loc[idx, "meas_best_s_m"] = float(dist[best_idx[k]])
                part.loc[idx, "pred_to_meas_best_m"] = float(abs(dist[pred_i] - dist[best_idx[k]]))
                part.loc[idx, "meas_margin_30m"] = float(margins[k])
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    for col in ["ll_at_pred", "pred_to_meas_best_m", "meas_margin_30m"]:
        med = float(out[col].median()) if np.isfinite(out[col]).any() else 0.0
        out[col] = out[col].fillna(med)
    return out


def feature_cols() -> list[str]:
    return [
        "pred_range",
        "to_median",
        "nearest_diff",
        "ll_at_pred",
        "pred_to_meas_best_m",
        "meas_margin_30m",
        "method_baseline",
        "method_speed",
        "method_total_gate",
        "method_axis_gate",
        "direction_forward",
    ]


def loso_predict(cand: pd.DataFrame) -> pd.DataFrame:
    outs = []
    feats = feature_cols()
    for test_seg in sorted(cand["segment_label"].unique()):
        train = cand[cand["segment_label"] != test_seg].copy()
        test = cand[cand["segment_label"] == test_seg].copy()
        clf = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=8,
            max_depth=5,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(train[feats], train["good"])
        prob = clf.predict_proba(test[feats])[:, 1]
        test["prob_good"] = prob
        try:
            auc = roc_auc_score(test["good"], prob)
        except ValueError:
            auc = np.nan
        test["segment_auc"] = auc
        outs.append(test)
    return pd.concat(outs, ignore_index=True)


def select_predictions(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = []
    oracle = []
    for sample_id, g in pred.groupby("sample_id"):
        best_prob = g.loc[g["prob_good"].idxmax()]
        selected.append(best_prob)
        best_err = g.loc[g["abs_error_m"].idxmin()]
        oracle.append(best_err)
    selected_df = pd.DataFrame(selected)
    oracle_df = pd.DataFrame(oracle)
    rows = []
    thresholds = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]
    for th in thresholds:
        sub = selected_df[selected_df["prob_good"] >= th]
        rows.append(metric_row(f"RF_LOSO_select_accept_p>={th:g}", sub, len(selected_df)))
    rows.append(metric_row("Oracle_best_method_upperbound", oracle_df, len(oracle_df)))
    for method in METHODS:
        m = pred[pred["method"] == method]
        rows.append(metric_row(method, m, len(m)))
    summary = pd.DataFrame(rows)
    return selected_df, summary


def metric_row(name: str, df: pd.DataFrame, total_count: int) -> dict[str, float | str]:
    if df.empty:
        return {
            "method": name,
            "accepted_count": 0,
            "total_count": total_count,
            "coverage_pct": 0.0,
            "median_abs_error_m": np.nan,
            "mean_abs_error_m": np.nan,
            "rmse_m": np.nan,
            "p75_abs_error_m": np.nan,
            "p90_abs_error_m": np.nan,
            "good_rate_25m": np.nan,
        }
    err = df["abs_error_m"].to_numpy(float)
    return {
        "method": name,
        "accepted_count": int(len(df)),
        "total_count": int(total_count),
        "coverage_pct": float(100 * len(df) / total_count),
        "median_abs_error_m": float(np.median(err)),
        "mean_abs_error_m": float(np.mean(err)),
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "p75_abs_error_m": float(np.percentile(err, 75)),
        "p90_abs_error_m": float(np.percentile(err, 90)),
        "good_rate_25m": float(np.mean(err <= GOOD_ERROR_M)),
    }


def plot(summary: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=180)
    view = summary[summary["method"].str.startswith("RF_LOSO")].copy()
    ax2 = ax1.twinx()
    ax1.plot(view["coverage_pct"], view["median_abs_error_m"], marker="o", label="median error")
    ax1.plot(view["coverage_pct"], view["p90_abs_error_m"], marker="s", label="p90 error")
    ax2.plot(view["coverage_pct"], view["good_rate_25m"], color="#2ca02c", marker="^", label="good rate <=25m")
    ax1.set_xlabel("Accepted coverage / %")
    ax1.set_ylabel("Accepted error / m")
    ax2.set_ylabel("Good rate")
    ax2.set_ylim(0, 1.02)
    ax1.set_title("LOSO reliability learning and rejection")
    ax1.grid(alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_notes(summary: pd.DataFrame, pred: pd.DataFrame, path: Path) -> None:
    aucs = pred.groupby("segment_label")["segment_auc"].first().reset_index()
    lines = [
        "# Reliability Learning Experiment",
        "",
        "Purpose: test whether method disagreement can be learned as an integrity/rejection layer.",
        "",
        "Protocol:",
        "",
        "- Candidate methods: Baseline total HMM, Speed-prior total HMM, TotalHP information-gated HMM, Axis-calibrated XY+Total mid-gated HMM.",
        "- Label: a candidate is good if its absolute error is <= 25 m.",
        "- Features: method identity, prediction range among methods, distance to median prediction, nearest-method disagreement, direction.",
        "- Validation: leave-one-segment-out over 5.13 segments. This is only a small-sample diagnostic, not yet a final publishable validation.",
        "",
        "Summary:",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Segment AUC:",
        "",
        aucs.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Interpretation:",
        "",
        "- If LOSO reliability improves accepted-sample median/P90 while retaining useful coverage, confidence/rejection is a promising route.",
        "- If it only works at tiny coverage, it is more suitable as an integrity monitor than a complete localization method.",
        "- A publishable version should train reliability on 4.14 leave-one-pass-out data and test on 5.13, or collect more days.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cand = build_candidate_table()
    pred = loso_predict(cand)
    selected, summary = select_predictions(pred)
    cand.to_csv(OUT_DIR / "reliability_candidates.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(OUT_DIR / "reliability_loso_predictions.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "reliability_selected_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "reliability_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "reliability_summary.json").write_text(
        json.dumps({"summary": summary.to_dict(orient="records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot(summary, OUT_DIR / "reliability_tradeoff.png")
    write_notes(summary, pred, OUT_DIR / "reliability_learning_notes.md")
    print(summary.round(3).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
