# Railway MagNav

铁路地磁导航数据处理与初步匹配验证代码。当前版本保留了良陈铁路小车实验中已经完成的流程：SPAN 位置数据与磁强计数据时间对齐、沿轨道方向 0.5 m 建图、往返方向处理、跨日期磁图相似度分析、滑窗 NCC 初步匹配验证，以及阶段性 Word 报告生成。

## 数据目录

默认数据目录为：

```text
C:\Users\<用户名>\Desktop\磁导航\数据\codex_railway_magnav
```

脚本默认读取：

```text
data/
  SPAN4.14/
  SPAN5.13/
  mag4.14/
  mag5.13/
```

默认输出到：

```text
data_proc_new/
```

原始数据、CSV、图片、Word 报告等产物不提交到 GitHub，只保留处理代码和说明文档。

## 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 处理流程

1. 建立 0.5 m 间隔磁图，并输出每趟对齐数据。

```powershell
python scripts\process_railway_magnav.py --write `
  --data-root "C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\data" `
  --out-dir "C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\data_proc_new"
```

2. 融合 4.14 和 5.13 磁图，计算相似度并画图。

```powershell
python scripts\analyze_magnetic_maps.py `
  --proc-dir "C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\data_proc_new"
```

3. 用 4.14 融合 total 磁图作为参考，验证 5.13 total 曲线滑窗匹配。

```powershell
python scripts\validate_magnetic_matching.py `
  --proc-dir "C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\data_proc_new"
```

4. 诊断 5.13 原始 SPAN 与磁强计时间覆盖关系。

```powershell
python scripts\diagnose_5_13_coverage.py `
  --data-root "C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\data" `
  --proc-dir "C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\data_proc_new"
```

5. 生成阶段性汇报 Word 文档。

```powershell
python scripts\build_stage_report_docx.py `
  --proc-dir "C:\Users\m1352\Desktop\磁导航\数据\codex_railway_magnav\data_proc_new"
```

## 当前方法要点

- SPAN 的 GPGGA 时间字段按 NovAtel/Hexagon OEM7 文档作为 UTC 处理，再加 8 小时转北京时间；默认不减闰秒。
- 4.14 和 5.13 共享同一条 PCA 拟合轨道轴和同一个物理 0 点。
- 每个地图点包含轨道距离、拟合坐标、插值 SPAN 坐标、各趟磁场数据，以及融合统计量。
- 往返方向会改变车体系三轴方向，因此对三轴分量先做每趟车体系中位数去偏置，再旋转到统一轨道坐标系，输出 `*_track_anom` 异常特征。
- 融合磁图默认使用每个距离点多趟观测的稳健中位数，同时保留均值、标准差和 MAD。
- 初步匹配 baseline 使用 total 场滑窗归一化互相关 NCC。

## 主要脚本

- `scripts/process_railway_magnav.py`：读取 SPAN GPGGA 和磁强计数据，完成时间对齐、轨道坐标拟合、方向处理和 0.5 m 建图。
- `scripts/analyze_magnetic_maps.py`：生成融合磁图、跨日期对比图、归一化对比图和相似度指标。
- `scripts/validate_magnetic_matching.py`：用 5.13 片段匹配 4.14 磁图，输出 NCC 匹配误差统计和示例图。
- `scripts/diagnose_5_13_coverage.py`：检查 5.13 SPAN 文件与磁强计文件的时间覆盖，解释宽表中空值来源。
- `scripts/build_stage_report_docx.py`：把关键表格和图片写入阶段性 Word 汇报。

下一步计划是在现有 baseline 上复现铁路磁图 Graph SLAM：用局部磁图和磁签名相关性生成 loop closure，再通过一维 pose graph 优化约束里程漂移。
