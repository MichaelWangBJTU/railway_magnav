from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from process_railway_magnav import (
    DATASETS,
    build_span_segments,
    prepare_global_span_geometry,
    read_dataset_mag,
)


OUT_DIR = Path(r"C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\data_proc_new")


def main() -> None:
    prepared, *_ = prepare_global_span_geometry(0)
    span_files = prepared["5.13"]
    segments = build_span_segments(span_files)
    mag = read_dataset_mag(DATASETS["5.13"])

    rows = []
    for sf in span_files:
        rows.append(
            {
                "type": "SPAN file",
                "name": sf.label,
                "start": sf.df["time"].min(),
                "end": sf.df["time"].max(),
                "s_min_m": sf.df["s_abs_m"].min(),
                "s_max_m": sf.df["s_abs_m"].max(),
                "mag_overlap_rows": "",
                "mag_files": "",
            }
        )
    for seg in segments:
        st = seg["time"].iloc[0]
        en = seg["time"].iloc[-1]
        m = mag[(mag["time"] >= st - pd.Timedelta(seconds=1)) & (mag["time"] <= en + pd.Timedelta(seconds=1))]
        rows.append(
            {
                "type": "SPAN segment",
                "name": seg["segment_label"].iloc[0],
                "start": st,
                "end": en,
                "s_min_m": seg["s_abs_m"].min(),
                "s_max_m": seg["s_abs_m"].max(),
                "mag_overlap_rows": len(m),
                "mag_files": ";".join(sorted(m["mag_label"].unique())),
            }
        )
    for label, g in mag.groupby("mag_label"):
        rows.append(
            {
                "type": "MAG file",
                "name": label,
                "start": g["time"].min(),
                "end": g["time"].max(),
                "s_min_m": "",
                "s_max_m": "",
                "mag_overlap_rows": len(g),
                "mag_files": label,
            }
        )

    diag = pd.DataFrame(rows).sort_values(["start", "type", "name"])
    diag.to_csv(OUT_DIR / "diagnose_5_13_time_coverage.csv", index=False, encoding="utf-8-sig")

    compact_cols = [
        "distance_m",
        "map_lat",
        "map_lon",
        "map_alt_m",
        "map_pass_count",
        "map_mag_x_track_anom_mean_nT",
        "map_mag_y_track_anom_mean_nT",
        "map_mag_z_track_anom_mean_nT",
        "map_mag_total_mean_nT",
        "map_mag_total_std_nT",
    ]
    wide = pd.read_csv(OUT_DIR / "magmap_5_13_0p5m.csv")
    wide[compact_cols].to_csv(OUT_DIR / "magmap_5_13_compact_0p5m.csv", index=False, encoding="utf-8-sig")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=160)
    y = 0
    labels = []
    colors = {"SPAN file": "#1f77b4", "SPAN segment": "#2ca02c", "MAG file": "#d62728"}
    for _, row in diag.iterrows():
        ax.barh(y, row["end"] - row["start"], left=row["start"], height=0.65, color=colors[row["type"]], alpha=0.85)
        text = row["name"]
        if row["type"] == "SPAN segment":
            text += f" | mag rows={row['mag_overlap_rows']}"
        labels.append(text)
        y += 1
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("北京时间")
    ax.set_title("5.13 SPAN 与磁强计原始数据时间覆盖诊断")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "diagnose_5_13_time_coverage.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
