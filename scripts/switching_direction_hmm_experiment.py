from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anchor_reference_hmm_experiment as arh
import axis_calibrated_hmm_experiment as hmm


OUT_DIR = Path(r"C:\Users\m1352\Documents\railway_magnav\switching_direction_hmm_experiment")
AXIS_VARIANT = "fwd_z_y_x_back_z_y_minusx"
SAMPLE_PERIOD = "4s"
STEP_M = 0.5


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def measurement_ll(q: hmm.QuerySegment, ref: dict[str, np.ndarray]) -> np.ndarray:
    return hmm.measurement_loglikelihood(
        q,
        ref,
        ["total_raw_hp_z"],
        {"total_raw_hp_z": 1.0},
        sigma=1.2,
        robust=True,
    )


def switching_viterbi(
    q: hmm.QuerySegment,
    ref: dict[str, np.ndarray],
    vmax_mps: float = 1.2,
    switch_penalty: float = 8.0,
    opposite_start_penalty: float = 12.0,
    edge_dwell_penalty: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    dist = ref["distance_m"]
    n_s = len(dist)
    n_t = len(q.time)
    ll = measurement_ll(q, ref)
    times = pd.to_datetime(q.time)
    ts = times.astype("int64").to_numpy(float) / 1e9
    dirs = np.array([-1, 1], dtype=int)  # backward, forward
    declared = 1 if q.direction == "forward" else -1
    dp = np.full((n_t, 2, n_s), -np.inf, dtype=np.float32)
    prev_dir = np.full((n_t, 2, n_s), -1, dtype=np.int8)
    prev_idx = np.full((n_t, 2, n_s), -1, dtype=np.int32)
    for di, d in enumerate(dirs):
        dp[0, di] = ll[0] - (0.0 if d == declared else opposite_start_penalty)

    edge_mask = (dist <= dist[0] + 5.0) | (dist >= dist[-1] - 5.0)

    for k in range(1, n_t):
        dt = max(0.2, float(ts[k] - ts[k - 1]))
        max_step = max(1, int(math.ceil(vmax_mps * dt / STEP_M)))
        for cdi, cd in enumerate(dirs):
            for j in range(n_s):
                if cd > 0:
                    lo = max(0, j - max_step)
                    hi = j + 1
                else:
                    lo = j
                    hi = min(n_s, j + max_step + 1)
                best_val = -np.inf
                best_pdi = -1
                best_i = -1
                for pdi, pd_ in enumerate(dirs):
                    cand = dp[k - 1, pdi, lo:hi].astype(float).copy()
                    if cand.size == 0:
                        continue
                    if pd_ != cd:
                        cand -= switch_penalty
                    rel = int(np.argmax(cand))
                    val = float(cand[rel])
                    if val > best_val:
                        best_val = val
                        best_pdi = pdi
                        best_i = lo + rel
                if best_i < 0:
                    continue
                moved = abs(j - best_i) * STEP_M
                speed = moved / max(dt, 1e-3)
                smooth_penalty = -0.025 * speed * speed
                dwell_penalty = -edge_dwell_penalty if edge_mask[j] and moved < 0.25 else 0.0
                dp[k, cdi, j] = best_val + smooth_penalty + dwell_penalty + ll[k, j]
                prev_dir[k, cdi, j] = best_pdi
                prev_idx[k, cdi, j] = best_i

    flat = int(np.argmax(dp[-1].reshape(-1)))
    cur_di, cur_i = np.unravel_index(flat, (2, n_s))
    path_idx = np.zeros(n_t, dtype=int)
    path_dir = np.zeros(n_t, dtype=int)
    path_idx[-1] = cur_i
    path_dir[-1] = dirs[cur_di]
    for k in range(n_t - 1, 0, -1):
        pdi = int(prev_dir[k, cur_di, cur_i])
        pi = int(prev_idx[k, cur_di, cur_i])
        if pdi < 0 or pi < 0:
            pdi = cur_di
            pi = cur_i
        cur_di, cur_i = pdi, pi
        path_idx[k - 1] = cur_i
        path_dir[k - 1] = dirs[cur_di]
    pred = dist[path_idx]
    switches = int(np.sum(path_dir[1:] != path_dir[:-1]))
    meta = {
        "switch_count": float(switches),
        "direction_forward_fraction": float(np.mean(path_dir > 0)),
        "final_direction": float(path_dir[-1]),
        "final_score": float(np.max(dp[-1])),
    }
    return pred, path_dir, meta


def fixed_total(q: hmm.QuerySegment, ref: dict[str, np.ndarray], vmax_mps: float) -> np.ndarray:
    pred, _ = hmm.viterbi_track(
        q,
        ref,
        ["total_raw_hp_z"],
        {"total_raw_hp_z": 1.0},
        sigma=1.2,
        vmax_mps=vmax_mps,
        robust=True,
        info_gate=False,
        start_prior="uniform",
    )
    return pred


def evaluate(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    err = pred - truth
    warmup = min(20, max(0, len(err) // 10))
    ew = err[warmup:]
    return {
        "median_abs_error_m": float(np.median(np.abs(ew))),
        "mean_abs_error_m": float(np.mean(np.abs(ew))),
        "rmse_m": float(np.sqrt(np.mean(ew * ew))),
        "p90_abs_error_m": float(np.percentile(np.abs(ew), 90)),
        "final_error_m": float(abs(err[-1])),
        "max_abs_error_m": float(np.max(np.abs(err))),
        "within_25m_rate": float(np.mean(np.abs(err) <= 25.0)),
        "within_50m_rate": float(np.mean(np.abs(err) <= 50.0)),
    }


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for method, g in rows.groupby("method"):
        out.append(
            {
                "method": method,
                "segment_count": int(len(g)),
                "median_abs_error_m": float(g["median_abs_error_m"].median()),
                "mean_abs_error_m": float(g["mean_abs_error_m"].mean()),
                "rmse_m": float(g["rmse_m"].mean()),
                "median_final_error_m": float(g["final_error_m"].median()),
                "mean_final_error_m": float(g["final_error_m"].mean()),
                "max_final_error_m": float(g["final_error_m"].max()),
                "mean_within_25m_rate": float(g["within_25m_rate"].mean()),
                "mean_within_50m_rate": float(g["within_50m_rate"].mean()),
            }
        )
    return pd.DataFrame(out).sort_values(["median_final_error_m", "rmse_m"])


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    methods = summary["method"].tolist()
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=180)
    ax.bar(x - 0.18, summary["median_abs_error_m"], 0.36, label="样本中位误差", color="#2a6fbb")
    ax.bar(x + 0.18, summary["median_final_error_m"], 0.36, label="终点中位误差", color="#2b9348")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("误差 / m")
    ax.set_title("方向切换 HMM：样本误差与终点误差")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    refs, _ = arh.build_candidate_refs()
    ref = refs["forward_only"]
    queries = hmm.read_query_segments(AXIS_VARIANT, SAMPLE_PERIOD)
    rows = []
    traj_rows = []
    switch_grid = [
        (1.2, 6.0, 0.00),
        (1.2, 10.0, 0.00),
        (1.4, 6.0, 0.00),
        (1.4, 10.0, 0.00),
        (1.2, 6.0, 0.02),
    ]
    for q in queries:
        candidates: list[tuple[str, np.ndarray, dict[str, float], np.ndarray | None]] = []
        for vmax in [1.0, 1.2, 1.4]:
            candidates.append((f"FixedTotal_vmax{vmax:g}", fixed_total(q, ref, vmax), {}, None))
        for vmax, sp, ep in switch_grid:
            pred, pdir, meta = switching_viterbi(q, ref, vmax_mps=vmax, switch_penalty=sp, edge_dwell_penalty=ep)
            candidates.append((f"SwitchTotal_vmax{vmax:g}_sw{sp:g}_edge{ep:g}", pred, meta, pdir))
        for method, pred, meta, pdir in candidates:
            rows.append({"segment_label": q.label, "direction": q.direction, "method": method, **evaluate(pred, q.truth_s), **meta})
            keep = np.linspace(0, len(pred) - 1, min(220, len(pred))).round().astype(int)
            for i in keep:
                traj_rows.append(
                    {
                        "segment_label": q.label,
                        "direction": q.direction,
                        "method": method,
                        "time": str(pd.Timestamp(q.time[i])),
                        "truth_s_m": float(q.truth_s[i]),
                        "pred_s_m": float(pred[i]),
                        "path_direction": float(pdir[i]) if pdir is not None else np.nan,
                    }
                )
    result = pd.DataFrame(rows)
    summary = summarize(result)
    traj = pd.DataFrame(traj_rows)
    result.to_csv(OUT_DIR / "switching_direction_hmm_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "switching_direction_hmm_summary.csv", index=False, encoding="utf-8-sig")
    traj.to_csv(OUT_DIR / "switching_direction_hmm_trajectories.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "switching_direction_hmm_summary.json").write_text(
        json.dumps({"summary": summary.to_dict(orient="records"), "results": result.to_dict(orient="records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_summary(summary, OUT_DIR / "switching_direction_hmm_summary.png")
    print(summary.round(3).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
