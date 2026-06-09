# 研究代码索引

本索引用于说明 `scripts/` 下主要脚本的作用。当前脚本以阶段性研究复现为主，部分脚本仍保留本机数据路径或实验输出目录。

## 数据处理与基础建图

- `process_railway_magnav.py`：读取 SPAN/GPGGA 和磁强计数据，完成时间对齐、沿轨坐标投影、0.5 m 磁图构建。
- `analyze_magnetic_maps.py`：融合 4.14/5.13 磁图，计算跨日相似度并绘制对比图。
- `validate_magnetic_matching.py`：total 场滑窗 NCC baseline。
- `diagnose_5_13_coverage.py`：诊断 5.13 时间覆盖和空值来源。
- `diagnose_senior_plot_axis_issue.py`：检查师兄图中轴向/尺度问题。
- `truth_axis_anomaly_diagnostic.py`：检查真值投影异常段。
- `verify_turnaround_and_trim.py`：检查反向尾段和单程单调区段评估。

## 磁匹配 baseline 与粗精匹配

- `msd_iccp_coarse_fine_check.py`：MSD + ICCP 粗精匹配检查。
- `axis_calibrated_full_matching.py`：轴校准后的全局匹配实验。
- `uniqueness_gate_analysis.py`：局部唯一性门限分析。
- `distance_warp_diagnostic.py`：DTW/距离弯曲诊断。
- `distance_warp_lopo_4_14.py`：4.14 组内留一可重复性验证。
- `sequence_hmm_experiment.py`：序列匹配与 HMM 初步实验。
- `latest_literature_aligned_experiments.py`：按最新文献路线做 top-k/SLAC-lite 等对齐实验。

## HMM/Viterbi 与 anchor map

- `axis_calibrated_hmm_experiment.py`：轴校准 HMM/Viterbi 主实现之一。
- `axis_calibrated_hmm_experiment.py` 相关输出目录：不同采样周期、门限、速度先验和轴变体实验。
- `anchor_reference_selection_experiment.py`：选择更稳定的 4.14 anchor reference。
- `constrained_map_alignment_experiment.py`：受约束地图对齐与参考图修正。
- `anchor_reference_hmm_experiment.py`：基于 anchor reference 的 HMM 实验。
- `forward_anchor_hmm_tuning.py`：前向 anchor HMM 参数调优，是当前最佳 selector 的总场候选来源。
- `plot_hmm_segment_diagnostics.py`：分段 HMM 诊断图。

## 弱 IMU 与无轮速实验

- `no_wheel_imu_experiments.py`：无轮速、磁匹配与 IMU 辅助的早期综合实验。
- `research_no_wheel_sota.py`：无轮速铁路地磁 SOTA 研究与复现实验骨架。
- `imu_progress_gated_ensemble.py`：使用 IMU 粗进度筛选候选。
- `endpoint_error_evaluation.py`：终点误差和覆盖率评估。
- `plot_imu_progress_review.py`：IMU 进度诊断图。
- `adaptive_speed_scale_experiment.py`：自适应速度尺度实验。
- `weak_mileage_sequence_filter.py`：弱里程序列匹配实验，当前结果为负。
- `fixed_lag_online_hmm_experiment.py`：固定滞后在线风格 HMM，当前结果较差。
- `coarse_start_online_hmm_experiment.py`：粗起点先验固定滞后 HMM。
- `endpoint_prior_hmm_experiment.py`：端点先验 HMM 检查，说明端点先验本身不足以解决重复磁特征。

## 候选选择与鲁棒性

- `progress_margin_selector_experiment.py`：当前最优方法，使用总场多 `vmax` 候选、轴候选、弱 IMU 进度和 Viterbi margin 选择最终轨迹。
- `delayed_multihypothesis_hmm_experiment.py`：延迟多假设 HMM，当前没有超过最佳 selector。
- `robust_candidate_scoring_experiment.py`：鲁棒候选评分，当前作为负结果和诊断。
- `reliability_learning_experiment.py`：可靠性学习与拒绝机制实验，受限于样本太少。
- `postprocess_confidence.py`：后处理置信度分析。
- `quality_gated_map_experiments.py`：质量门控磁图实验。

## 方向与尾段处理实验

- `switching_direction_hmm_experiment.py`：自由方向切换 HMM，当前结果变差。
- `signed_imu_prior_hmm_experiment.py`：全局 signed IMU 运动先验 HMM，当前结果变差。
- `imu_switch_signed_suffix_experiment.py`：只在检测到尾段时切换 signed IMU，仍不足以稳定解决终点问题。
- `imu_direction_piecewise_hmm.py`：分段方向/IMU 运动实验。

## 文献复现和报告生成

- `reproduce_sota_methods.py`：早期 SOTA 复现实验脚本。
- `build_sota_repro_report.py`：SOTA 复现报告。
- `build_no_wheel_sota_report.py`：无轮速 SOTA 报告。
- `build_no_wheel_imu_report.py`：无轮速 IMU 报告。
- `build_latest_literature_stage_report.py`：最新文献对齐实验报告。
- `build_imu_progress_update_report.py`：IMU 进度更新报告。
- `build_anchor_hmm_stage_report.py`：anchor HMM 阶段报告。
- `build_method_record_docx.py`：方法记录 Word。
- `build_stage_report_docx.py`：早期阶段报告。
- `build_supervisor_brief_20260609.py`：导师汇报报告生成脚本。

## 精选结果文件

`docs/results/` 中保留少量小型结果文件：

- `progress_margin_selector_summary.csv`：当前最佳方法汇总指标。
- `progress_margin_selector_results.csv`：分段指标。
- `progress_margin_selector_decisions.csv`：每段最终选择的候选。
- `progress_margin_selector_selected_trajectories.csv`：当前最佳轨迹图使用的黑线/橙线数据。
- `turnaround_and_trim_summary.json`：反向尾段和单程单调区段评估。
- `endpoint_prior_hmm_summary.csv`：端点先验负结果。
- `distance_warp_lopo_4_14_summary.csv`：4.14 组内 DTW 可重复性。
