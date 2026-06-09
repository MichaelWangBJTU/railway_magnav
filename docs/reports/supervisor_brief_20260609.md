# 铁路地磁定位阶段性技术路线与方向建议

日期：2026-06-09  
项目：良陈铁路约 700 m 小车地磁采集，4.14 建图，5.13 跨日验证

## 核心结论

1. 当前数据证明铁路沿线地磁特征具有可重复性，但目前还不能支撑“达到铁路地磁定位 SOTA”的结论。
2. 当前最优方法为 Progress-Margin Selector：多 vmax 总场 HMM 候选 + 轴校准候选 + 弱 IMU 进度与 Viterbi margin 筛选。5 个可用 5.13 段的原始全段 RMSE 为 40.1 m，中位误差 13.8 m；单程单调区段 RMSE 为 39.0 m，平均终点误差 16.7 m。
3. 与代表性铁路 SOTA（1 m 到 5 m 级）差距明显，主要差在数据平台、里程/速度约束、地图规模、多日数据量、在线完整性判断，而不是“地磁不可用”。
4. 小车采集铁路地磁数据可行，但当前 700 m、两天、无轮速计的数据只能支撑可行性验证和方法雏形；若要发三区论文，建议至少补到 4 天、20 条以上单程、3 个以上跨日验证组合。
5. 后续方向建议：短期继续铁路，作为受约束的一维场景把方法做实；同时准备城市车辆平台作为备选或下一阶段扩展。若导师更看重应用面和数据规模，城市车辆平台更有发展空间；若更看重可控性和快速形成论文闭环，铁路方向仍值得再做一轮高质量采集。

## 最主要参考文献

最主要参考文献建议放 Siebler, Heirich, Sand, *Train Localization with Particle Filter and Magnetic Field Measurements*, FUSION 2018。原因是它和本项目最接近：都是铁路沿轨一维地磁地图定位，都用已有磁图和运动模型约束位置估计，并且其 SIR 粒子滤波报告沿轨 RMSE 3.84 m，是当前“无轮速/弱运动约束铁路地磁定位”最重要的公平基线。

强 SOTA 对标文献再补两篇：FUSION 2024 *Magnetic Field Mapping of Railway Lines with Graph SLAM*，以及 2025 arXiv *Real-time rail vehicle localisation using spatially resolved magnetic field measurements*。

## 当前指标最好方法怎么做

当前指标最好的方法叫 Progress-Margin Selector。它不是单独用一条磁曲线硬匹配，而是先生成多条候选轨迹，再根据弱 IMU 进度和匹配置信度选择最可信的一条。

1. 用 4.14 数据建立参考磁图，主特征用总场高通值，原因是总场不受小车朝向翻转影响，比三轴更稳定。
2. 对 5.13 每个查询段做 HMM/Viterbi 匹配。HMM 状态是沿轨位置 s_t，转移约束是相邻时刻不能跳太远，观测似然来自当前位置磁场与磁图是否相似。
3. 不只跑一个速度上限，而是分别跑 vmax=1.0、1.2、1.4 m/s，得到多条总场候选轨迹，用来处理小车速度不稳定的问题。
4. 另外跑一条轴校准候选。三轴磁数据有时能补充总场，但跨日不够稳定，所以只作为候选，不作为默认主方法。
5. 对每条候选计算 candidate_progress，再与 INSPVAX 积分得到的 imu_progress 比较：progress_compat = |log((candidate_progress + 10)/(imu_progress + 10))|。数值越小，说明这条候选轨迹的总进度越像真实运动。
6. 在总场候选里，先保留 progress_compat 接近最优的候选，再看 Viterbi final_score_margin；如果仍然相近，偏向更大的 vmax，避免末端跟不上。
7. 只有当轴校准候选的 Viterbi margin 足够高，并且进度一致性明显优于总场候选时，才切换到轴候选。
8. SPAN/GPGGA 真值只用于最后算误差，不参与候选选择。
