# 无轮速计铁路地磁定位方法记录

本文档记录当前阶段已经验证过的方法链条、公式、实验结论和可写入论文的候选创新点。记录原则是：只保留有文献依据或本数据集实验证据支撑的方法；负结果也记录，避免后续重复试错。

## 1. 场景与约束

实验场景为良陈铁路约 700 m 区段的轨道小车地磁采集。当前可用数据包括：

- 三轴磁强计：`mag_x, mag_y, mag_z, mag_total`。
- SPAN/GNSS 真值：用于建图和评价。
- SPAN/INS 日志：可用 INSPVAX / BESTVEL 等，但当前没有轮速计或里程计。

核心限制：

- 不能直接使用铁路 Graph SLAM 2024 的完整输入条件，因为该方法依赖 odometer / wheel encoder。
- 无轮速条件下，合理 baseline 应是磁图 + 磁强计 + 运动模型的递归贝叶斯定位，例如 FUSION 2018 的粒子滤波。
- IMU/INS 可以作为弱先验，但当前 INSPVAX 水平速度积分在不同片段尺度不稳定，不能直接替代轮速计。

## 2. 文献定位

当前与本项目条件最接近的方法族如下：

1. Siebler et al., FUSION 2018, *Train Localization with Particle Filter and Magnetic Field Measurements*。
   - 输入：轨道磁图、车载磁强计。
   - 状态：沿轨位置和速度。
   - 方法：SIR 粒子滤波。
   - 与本项目关系：最适合作为无轮速计铁路磁定位 baseline。

2. Siebler et al., PLANS 2020, *Joint Train Localization and Track Identification based on Earth Magnetic Field Distortions*。
   - 输入：磁强计、加速度计、地图。
   - 方法：递归贝叶斯滤波，同时做定位和轨道识别。
   - 与本项目关系：后续若扩展多轨/道岔，可参考。

3. Siebler et al., ION GNSS+ 2022, *Robust Particle Filter for Magnetic Field-based Train Localization*。
   - 核心：用 LRT 检测异常磁观测，避免磁干扰拖垮粒子滤波。
   - 与本项目关系：支持“重尾鲁棒似然”和“匹配置信度门控”的设计。

4. Siebler et al., FUSION 2024, *Magnetic Field Mapping of Railway Lines with Graph SLAM*。
   - 输入：磁强计 + odometer。
   - 方法：每个节点保存局部磁图，用磁场相关做 loop closure，加入 pose graph。
   - 与本项目关系：强 SOTA 参考，但当前无轮速计条件下不能直接作为公平 baseline。

5. Dieckow et al., arXiv 2025, *Real-time rail vehicle localisation using spatially resolved magnetic field measurements*。
   - 核心：重尾粒子滤波 + 序列对齐 + 混合初始化。
   - 与本项目关系：最新方向参考，但其硬件为更强的空间分辨磁测量，不能直接等价比较。

## 3. 数据预处理

### 3.1 时间与坐标

SPAN GPGGA 的 NMEA 时间按 UTC 理解，再转换为北京时间。NovAtel ASCII header 中的 GPS week / seconds-of-week 属于 GPS 时间，需要：

```text
t_UTC = t_GPS - leap_seconds
t_BJ = t_UTC + 8 h
```

当前 leap seconds 使用 18 s。

SPAN 位置点投影到一条 PCA 拟合的轨道轴，得到沿轨绝对距离：

```text
s_abs = dot([east, north] - origin, rail_axis)
```

磁强计数据按时间与 SPAN 位置对齐，再映射到 0.5 m 间隔的距离网格。

### 3.2 方向处理

小车往返时掉头，车体系三轴方向会随行驶方向变化。当前做法：

- 以轨道方向建立 track frame。
- 对每趟去除 body-frame 中值，构造异常量。
- 对方向翻转后的分量进行一致化，主要使用：
  - `mag_total`
  - `mag_x_track_anom`
  - `mag_y_track_anom`
  - `mag_z_track_anom`

实验显示跨日场景下 `total` 和部分高通特征最稳定，`y` 特征在某些配置有效但并不总是提升。

## 4. 特征构造

### 4.1 总场标准化

对每段查询磁序列和参考磁图分别做 robust z-score：

```text
z(x) = clip((x - median(x)) / (1.4826 * MAD(x)), -6, 6)
```

这样可以降低跨日期绝对强度偏置和局部异常值的影响。

### 4.2 总场高通

为了保留局部磁异常形状、削弱慢变背景和传感器偏置，使用滑动中值高通：

```text
m_HP(s) = m(s) - median{m(u) | |u - s| <= W/2}
```

当前距离域参考图使用 `W = 40 m`；时间域查询序列使用约 60 s 的滑动中值窗口，再进行 robust z-score。

当前最佳跨日方法使用特征：

```text
F = {total_z, total_hp_z}
```

其中：

- `total_z` 保留较长尺度的磁图轮廓。
- `total_hp_z` 强调局部可复现磁异常。

## 5. 无轮速计 baseline：磁场递归贝叶斯 / 粒子滤波

### 5.1 状态模型

沿轨状态：

```text
x_k = [s_k, v_k]^T
```

其中 `s_k` 为沿轨位置，`v_k` 为速度。无轮速计条件下，速度不是观测量，而是状态的一部分。

简化运动模型：

```text
v_k = clip(v_{k-1} + noise, 0, v_max)
s_k = s_{k-1} + d * v_k * Δt + noise
```

其中 `d` 为方向，forward 时 `d=+1`，backward 时 `d=-1`。

### 5.2 观测模型

FUSION 2018-style baseline 使用总场单特征：

```text
r_k(s) = z_total,k - M_total(s)
log p(z_k | s) = -0.5 * (r_k(s) / σ)^2
```

其中 `M_total(s)` 是 4.14 参考磁图在位置 `s` 的特征值。

### 5.3 粒子滤波

实现了 SIR particle filter：

1. 初始化粒子位置 `s` 在轨道范围内均匀分布。
2. 按运动模型预测 `s, v`。
3. 按磁场观测似然更新权重。
4. 当有效粒子数 `N_eff` 低于阈值时系统重采样。

在当前数据上，`SOTA2018_PF_total` 的 5.13 跨日中位误差为 `156.1 m`。

## 6. 离散 HMM / Viterbi 版本

为了减少粒子滤波随机性，并在短线路上得到更稳定的全局最优路径，实现了离散 HMM / Viterbi。

### 6.1 状态离散化

将轨道位置离散为 0.5 m 网格：

```text
s_j = 0.5 * j
```

### 6.2 转移约束

由方向和最大速度限制可达状态：

```text
0 <= d * (s_j - s_i) <= v_max * Δt
```

不可达转移的概率置为 0。

### 6.3 Viterbi 递推

设 `D_k(j)` 为第 `k` 个时刻到达位置 `s_j` 的最大 log posterior：

```text
D_k(j) = log p(z_k | s_j) + max_i [D_{k-1}(i) + log p(s_j | s_i)]
```

并保存回溯指针，最终得到整段最优沿轨位置序列。

在当前数据上，`SOTA2018_Viterbi_total` 的 5.13 跨日中位误差为 `102.8 m`，优于随机粒子滤波 baseline。

## 7. 当前最优改进方法：RobustTotalHP-Viterbi

### 7.1 方法定义

当前最优方法命名为：

```text
Proposed_RobustTotalHP_Viterbi
```

它相对于 SOTA2018-style Viterbi 改动两点：

1. 多尺度总场特征：`total_z + total_hp_z`。
2. 重尾鲁棒似然：用 Student-t-like loss 替代高斯 loss。

### 7.2 重尾鲁棒似然

对每个特征 `f`：

```text
r_f(k, j) = z_f,k - M_f(s_j)
```

高斯似然：

```text
log p_f = -0.5 * (r_f / σ)^2
```

鲁棒似然：

```text
log p_f = -0.5 * (ν + 1) * log(1 + (r_f / σ)^2 / ν)
```

当前使用 `ν = 3`。

多特征加权：

```text
log p(z_k | s_j) = Σ_f w_f log p_f(z_f,k | s_j)
```

当前最佳权重：

```text
w_total_z = 0.7
w_total_hp_z = 1.0
```

### 7.3 实验效果

5.13 查询、4.14 原始融合参考图：

| 方法 | 中位绝对误差 | 平均 RMSE |
|---|---:|---:|
| SOTA2018_PF_total | 156.1 m | 167.4 m |
| SOTA2018_Viterbi_total | 102.8 m | 155.2 m |
| Proposed_RobustTotalHP_Viterbi | 62.8 m | 123.1 m |

因此，在当前数据集上，该方法超过了复现的无轮速计 SOTA-style baseline。

注意：这不是说已经达到文献米级 SOTA。当前数据条件为短线路、单磁强计、跨日期、无轮速计，难度和文献条件不同。

## 8. 4.14 组内留一验证

为了验证初始预处理是否可靠，做了 4.14 组内 leave-one-pass-out (LOPO) 验证。

### 8.1 距离域 LOPO

对每一趟 `q`：

1. 从参考图构建中排除 `q`。
2. 用其它趟的中值磁图作为参考图。
3. 在距离域滑动查询片段，最大化 `total + total_hp` 的相关性。

相似度：

```text
score(p) = mean( corr(q_total, M_total(p + rel)),
                 corr(q_HP, M_HP(p + rel)) )
```

组内距离域 LOPO 结果：

```text
median abs error = 0.5 m
```

这说明时间对齐、方向一致化、距离投影和 0.5 m 地图映射整体可靠。

### 8.2 异常片段

两个 4.14 backward 片段异常：

- `BMAW15230010L_1_seg02`：距离域误差 `96.0 m`
- `BMAW15230010L_1_seg03`：距离域误差 `184.5 m`，margin `0.006`

诊断图显示异常片段存在尖峰、削顶/饱和和重复特征。它们不代表整体预处理失败，但提示后续需要数据质量评价。

## 9. 质量门控参考图实验

### 9.1 硬门控

基于 4.14 组内 LOPO 结果定义质量门控：

```text
pass if abs_error <= 2 m and best_score >= 0.60
```

被拒绝片段：

- `BMAW15230010L_1_seg02`
- `BMAW15230010L_1_seg03`

使用通过门控的 7 个片段重建 4.14 参考图。

结果：硬门控没有提升跨日中位误差。对 `RobustTotalHP`，中位误差从 `62.8 m` 变为 `88.7 m`，但 RMSE 从 `123.1 m` 降到 `116.0 m`。

解释：硬删除异常趟减少了局部污染，但也损失了跨日代表性和空间覆盖。

### 9.2 软门控

为了避免硬删除，定义软权重：

```text
err_score = exp(-(abs_error / 50)^2)
corr_score = clip((best_score - 0.35) / 0.35, 0, 1)
quality_weight = 0.15 + 0.85 * err_score * corr_score
```

然后用加权中值重建参考图。

结果：

- `SOTA2018_Viterbi_total`：中位误差从 `102.8 m` 降到 `82.8 m`。
- `RobustTotalHP`：中位误差为 `83.9 m`，不如原始融合图的 `62.8 m`，但平均误差/RMSE 略低。

结论：质量信息适合作为鲁棒权重或置信度特征，但不能简单认为“组内异常越少，跨日越好”。

## 10. 当前可写论文的候选创新点

### 创新点 A：无轮速计铁路场景的鲁棒总场-高通 HMM 定位

定义：

```text
RobustTotalHP-Viterbi = one-dimensional HMM/Viterbi
                      + total_z
                      + total_hp_z
                      + Student-t robust likelihood
```

支撑：

- 文献依据：FUSION 2018 粒子滤波、2022 鲁棒 PF。
- 实验证据：5.13 跨日中位误差从 SOTA-style Viterbi baseline 的 `102.8 m` 降至 `62.8 m`。

这是当前最稳的可写方法。

### 创新点 B：组内 LOPO 一致性作为磁图质量评价，而非简单筛选

定义：

```text
quality(segment) = function(LOPO_error, best_score, score_margin)
```

作用：

- 发现异常趟。
- 作为后续地图构建权重或定位置信度。
- 不建议直接硬删除所有异常趟。

支撑：

- 4.14 组内 LOPO 中位误差 0.5 m，能识别两个明显异常片段。
- 硬门控跨日中位误差变差，说明质量门控必须软化或结合跨日代表性。

### 创新点 C：从“匹配结果”转向“置信可拒绝定位”

当前许多失败来自磁特征自相似。实际系统不应该每个时刻都强行输出位置，而应输出：

```text
position estimate + confidence / reject flag
```

可用置信度包括：

- Viterbi posterior gap
- best-vs-second magnetic score margin
- map feature information content
- LOPO-derived local map reliability

这一点与铁路磁定位中的 loop closure 验证和鲁棒 PF 文献一致。

## 11. 下一步建议

1. 转换 `RAWIMUSX` 或等价原始 IMU 数据。
2. 做沿轨方向 IMU 预积分，而不是直接用 INSPVAX 速度积分。
3. 在 HMM / factor graph 中加入：
   - IMU 方向/加速度弱约束
   - 速度非负和最大速度约束
   - 磁观测鲁棒因子
   - LOPO 地图可靠性权重
4. 将定位输出改为“位置 + 置信度 + 可拒绝判断”。

## 12. 当前关键文件

- `research_no_wheel_sota.py`：无轮速计 SOTA-style baseline 与 RobustTotalHP 方法。
- `validate_4_14_intra_day.py`：4.14 组内 LOPO 验证。
- `quality_gated_map_experiments.py`：硬/软质量门控参考图实验。
- `no_wheel_sota/outputs/no_wheel_sota_summary.csv`：5.13 跨日核心结果。
- `no_wheel_sota/intra_day_4_14/outputs/intra_4_14_lopo_summary.csv`：4.14 组内验证结果。
- `no_wheel_sota/quality_gated_map/outputs/original_vs_quality_ref_summary.csv`：原始参考图、硬门控、软门控对比。

