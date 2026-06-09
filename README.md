# Railway MagNav

铁路地磁导航数据处理、建图、匹配验证和阶段性研究代码。

本仓库保存良陈铁路约 700 m 小车实验的主要处理流程和研究实验脚本。当前数据流程以 `4.14` 采集日作为参考磁图，以 `5.13` 采集日作为跨日查询验证，重点研究无轮速计条件下的铁路一维地磁地图匹配。

## 当前研究状态

已经实现并记录的主要内容：

- SPAN/GPGGA 与磁强计数据时间对齐；
- GNSS 坐标投影到沿轨一维距离 `s`；
- 0.5 m 间隔铁路磁图构建；
- 往返方向、车体系三轴磁场、总场磁场的预处理；
- 跨日磁图相似度分析；
- 滑窗 NCC/MSD、DTW、HMM/Viterbi、弱 IMU 进度约束、多候选选择等 baseline 和改进方法；
- 当前最优的 `ProgressMarginSelector` 实验；
- 固定滞后 HMM、弱里程、端点先验、方向切换、鲁棒候选打分等负结果实验；
- 文献对标、SOTA 分析和导师汇报材料。

当前最优离线/延迟候选选择结果：

| 方法 | 评价集 | 中位误差 | 均值误差 | RMSE | 平均终点误差 |
|---|---|---:|---:|---:|---:|
| ProgressMarginSelector | 5.13 对 4.14，原始全段 | 13.8 m | 25.2 m | 40.1 m | 24.6 m |
| ProgressMarginSelector | 单程单调区段 | 13.8 m | 24.3 m | 39.0 m | 16.7 m |
| FixedTotal HMM baseline | 5.13 对 4.14，原始全段 | 24.6 m | 46.0 m | 51.2 m | 50.6 m |

这些结果说明铁路磁特征具有跨日可重复性，但目前还不能宣称达到铁路磁定位 SOTA。当前主要差距在数据规模、里程/速度约束、实时性、重复磁特征歧义和多日泛化验证。

## 仓库结构

```text
scripts/
  数据处理、建图、baseline、HMM/PF 风格实验、报告生成脚本

docs/research_trace/
  阶段性研究记录、文献矩阵、投稿策略和方法记录

docs/results/
  精选小体量结果 CSV/JSON，用于复查关键指标和图中数据

docs/figures/
  精选图表：当前最佳方法、轨迹图、SOTA 对比图等

docs/reports/
  导师汇报材料的 Markdown 版本
```

原始 SPAN、磁强计数据、大型对齐样本、完整生成产物不提交到 GitHub。

## 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 常用流程

默认本地数据目录：

```text
C:\Users\<user>\Desktop\磁导航\数据\codex_railway_magnav
```

### 1. 数据对齐与 0.5 m 建图

```powershell
python scripts\process_railway_magnav.py --write `
  --data-root "C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\data" `
  --out-dir "C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\data_proc_new"
```

### 2. 跨日磁图分析

```powershell
python scripts\analyze_magnetic_maps.py `
  --proc-dir "C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\data_proc_new"
```

### 3. 滑窗 NCC baseline

```powershell
python scripts\validate_magnetic_matching.py `
  --proc-dir "C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\data_proc_new"
```

### 4. 当前最佳候选选择方法

```powershell
python scripts\progress_margin_selector_experiment.py
```

输出的关键文件：

- `docs/results/progress_margin_selector_summary.csv`
- `docs/results/progress_margin_selector_results.csv`
- `docs/results/progress_margin_selector_decisions.csv`
- `docs/results/progress_margin_selector_selected_trajectories.csv`

## 当前最佳方法简述

`ProgressMarginSelector` 的核心思想是：先生成多条可能的沿轨位置轨迹，再用弱 IMU 进度和 Viterbi 匹配置信度选择最可信的一条。

1. 用 4.14 构建参考磁图，主特征使用总场高通值；
2. 对 5.13 每个查询段运行 HMM/Viterbi；
3. 用 `vmax = 1.0 / 1.2 / 1.4 m/s` 生成多条总场候选；
4. 额外生成一条轴校准候选；
5. 比较候选轨迹总进度与 INSPVAX 积分粗进度；
6. 在进度一致性较好的候选中，再根据 Viterbi final score margin 选择；
7. 只有当轴候选匹配置信度和进度一致性都明显优于总场候选时才切换到轴候选；
8. SPAN/GPGGA 真值只用于最终评价，不参与候选选择。

## 主要参考文献

主参考 baseline：

- Siebler, Heirich, Sand, *Train Localization with Particle Filter and Magnetic Field Measurements*, FUSION 2018. https://elib.dlr.de/119898/1/FUSION_2018.pdf

强 SOTA 对标：

- Siebler et al., *Magnetic Field Mapping of Railway Lines with Graph SLAM*, FUSION 2024. https://isas.iar.kit.edu/pdf/FUSION24_Siebler.pdf
- Dieckow et al., *Real-time rail vehicle localisation using spatially resolved magnetic field measurements*, arXiv 2025. https://arxiv.org/abs/2507.19327

## 注意

当前仓库中的脚本以研究复现和阶段性实验为主，部分路径仍然保留本机数据目录或实验输出目录。后续若要整理成可发布软件包，需要进一步参数化数据路径、抽象公共模块、统一配置文件，并补充测试数据。
