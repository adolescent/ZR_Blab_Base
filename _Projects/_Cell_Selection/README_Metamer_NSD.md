# Metamer_NSD 细胞筛选与导出 — 数据结构与使用说明

从 `Metamer_NSD` 刺激集的 site class 中筛选细胞，导出五脑区（ML、MSB、AL、ASB、ALO）的标准化响应数据。流程与 `Metamer1k_Summary.py` 一致，仅刺激集与图像数量不同。

**导出脚本**：`MetamerNSD_Summary.py`

**默认保存目录**：

`E:\#Preprocessed_Data\Selected_Cells\Metamers\Metamer_NSD_2k`

---

## 刺激集布局

`Metamer_NSD` 刺激集共 **2216** 张图（见 `Py_Structure/Info_Files/Metamer_NSD.tsv`）：

| 原始 stim 索引 | 数量 | 内容 | 是否导出 |
|----------------|------|------|----------|
| 0–71 | 72 | FOB72（调谐计算） | 仅 FOB 导出 |
| 72–215 | 144 | FOB 重复块 | 否 |
| 216–1215 | 1000 | Metamer | 是，列 0–999 |
| 1216–2215 | 1000 | NSD1k | 是，列 1000–1999 |

导出列与原始 stim 索引关系：

```
export_col = raw_stim_index - 216
```

切片（与脚本一致）：

```python
SLICE_METAMER = slice(0, 1000)      # 1000 metamer
SLICE_NSD     = slice(1000, 2000)   # 1000 NSD
N_IMG = 2000
```

---

## 目录结构

```
Metamer_NSD_2k/
├── README_Metamer_NSD.md          # 本说明
├── stim_layout.npz                # 刺激索引映射（各脑区共享）
├── ML/
├── MSB/
├── AL/
├── ASB/
└── ALO/
    ├── avr_rsp.npy
    ├── psth.npy
    ├── trials_raw.npy
    ├── trials_raw_meta.npz
    ├── trials_raw_binned5ms.npz
    ├── trials_rsp.npz
    ├── cell_site_info.csv
    ├── cell_site_info.joblib
    ├── site_manifest.joblib
    ├── fob_avr.npy
    ├── fob_by_trial.npz
    ├── fob_meta.npz
    ├── heatmap_2k.png
    ├── heatmap_fob.png
    └── raster_first40.png
```

---

## 根目录共享文件

### `stim_layout.npz`

| 字段 | shape | 说明 |
|------|-------|------|
| `data_ids` | (2000,) | 原始 stim 索引（216–2215） |
| `stim_index` | (2000,) | 导出列索引（0–1999） |
| `stim_set` | (2000,) | TSV 中 `Stim_Set` 列 |
| `category` | (2000,) | TSV 中 `Category` 列 |
| `object_id` | (2000,) | TSV 中 `Object` 列 |
| `stim_type` | (2000,) | `'metamer'` / `'nsd'` |
| `slice_metamer` | (2,) | `[0, 1000)` |
| `slice_nsd` | (2,) | `[1000, 2000)` |
| `n_img` / `n_metamer` / `n_nsd` | scalar | 2000 / 1000 / 1000 |

---

## 各脑区文件（`{area}/`）

### 响应数组

| 文件 | shape | 说明 |
|------|-------|------|
| `avr_rsp.npy` | `(n_cell, 2000)` | 50–219 ms 窗口内 spike count，trial 平均 |
| `psth.npy` | `(n_cell, 2000, 450)` | 全时间 PSTH，trial 平均（时间轴 -100–349 ms，1 ms bin，onset=300） |
| `trials_raw.npy` | `(n_cell, n_repeat_max, 2000, 450)` | trial 级原始 spike count，不足 repeat 处为 NaN |
| `trials_rsp.npz` | `trials_rsp`: `(n_cell, n_repeat_max, 2000)` | trial 级 50–219 ms 窗口响应 |
| `trials_raw_binned5ms.npz` | `trials`: `(n_cell, n_repeat_max, 2000, 90)` | 5 ms 分箱的 trial 数据 |

### 细胞元信息

| 文件 | 说明 |
|------|------|
| `cell_site_info.csv` / `.joblib` | 每细胞一行：`global_idx`, `site_name`, `date`, `subject`, `local_cell_idx`, `stimset`, `dprime_face`, `dprime_body`, `ceiling_index`, `n_repeat` |
| `site_manifest.joblib` | 按 recording site 的轻量索引（含 `mtime`），供增量更新与独立 FOB 重跑 |

### FOB 调谐

| 文件 | shape | 说明 |
|------|-------|------|
| `fob_avr.npy` | `(n_cell, 150)` | FOB72 平均响应（有效长 72，其余 NaN padding） |
| `fob_by_trial.npz` | `fob_by_trial`: `(n_cell, n_repeat_max, 150)` | trial 级 FOB |
| `fob_meta.npz` | `fob_valid_len`, `fob_style` | 每细胞有效 FOB 长度（72）与格式（`FOB72`） |

### 质控图

| 文件 | 说明 |
|------|------|
| `heatmap_2k.png` | 2000 刺激 per-neuron z-score 热图 |
| `heatmap_fob.png` | FOB72 per-neuron z-score 热图 |
| `raster_first40.png` | 前 40 张刺激的平均 PSTH raster（5 ms bin） |

---

## 筛选规则

| 参数 | 值 | 说明 |
|------|-----|------|
| `CEILING_THRES` | 0.3 | noise ceiling 阈值 |
| `DP_THRES` | 0.5 | FOB D' 阈值 |
| `TIME_SLICE` | 150:320 | 响应窗口（50–219 ms） |
| `MAX_REPEAT` | 20 | trial 维上限 |

### 各脑区偏好类别（`AREA_PREFER`）

| 脑区 | 偏好 | 筛选说明 |
|------|------|----------|
| ML | Face | FOB Face D' > 0.5 |
| AL | Face | FOB Face D' > 0.5 |
| MSB | Body | FOB Body D' > 0.5 |
| ASB | Body | FOB Body D' > 0.5 |
| **ALO** | **all** | **仅 noise ceiling > 0.3，不按 FOB 刺激偏好筛选** |

仅处理 `stimset == 'Metamer_NSD'` 的 site class 文件。

---

## 增量导出

脚本默认**不会**每次重跑全部 site，而是按 site 的 joblib `mtime` 判断是否需要更新：

| 情况 | 行为 |
|------|------|
| site 未变 | 从已保存的 `avr_rsp` / `psth` / `trials_raw` 复用，不加载 joblib |
| 新增 / 修改 / 删除 site | 仅处理有变化的 site，并重建该脑区输出 |
| 某脑区完全无变化 | 跳过该脑区（打印 `no site changes, skip export`） |
| FOB 导出 | 仅对本次有更新的脑区运行 |

`site_manifest.joblib` 中每条记录包含 `path`、`mtime`、`selected`、`offset` 等字段，用于增量判断。

**首次添加 ALO**：已有 ML/MSB/AL/ASB 数据若无 site 变化会自动跳过，仅导出 `ALO/` 子目录。

---

## 运行方式

### 0. 更新 site-class 索引（首次或新增 site 后）

```bash
cd _Projects/_Cell_Selection
python Site_Class_Lite_Scan.py
```

索引保存于：

`E:\#Preprocessed_Data\SiteClass\Metamers\site_class_lite_index.joblib`（及同名 `.csv`）

扫描为增量式：仅重新解析 mtime/size 变化的文件。Summary 脚本每次运行也会对索引做轻量 refresh，自动收录 ALO 等新目录下的 site。

### 1. 导出 Metamer_NSD 数据

在 IDE 分 cell 运行 `MetamerNSD_Summary.py`，或：

```bash
cd _Projects/_Cell_Selection
python MetamerNSD_Summary.py
```

### 开关

| 开关 | 默认 | 说明 |
|------|------|------|
| `RUN_LITE_SCAN` | `False` | `True` 时显示完整索引扫描进度；否则静默增量 refresh |
| `RUN_SITE_REFRESH` | `False` | 一次性刷新 site class（MF→ML、重算 ceiling/FOB） |
| `RUN_FOB_EXPORT` | `True` | 主流程后导出 FOB（仅更新过的脑区） |

### 数据路径

| 路径 | 用途 |
|------|------|
| `E:\#Preprocessed_Data\SiteClass\Metamers\ML_MSB` | ML / MSB site class |
| `E:\#Preprocessed_Data\SiteClass\Metamers\AL_ASB` | AL / ASB site class |
| `E:\#Preprocessed_Data\SiteClass\Metamers\ALO` | ALO site class |
| `Py_Structure/Info_Files/Metamer_NSD.tsv` | 刺激元信息 |

---

## 读取示例

```python
import numpy as np
import pandas as pd

root = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Metamer_NSD_2k'
area = 'ALO'

avr = np.load(f'{root}/{area}/avr_rsp.npy')          # (n_cell, 2000)
info = pd.read_csv(f'{root}/{area}/cell_site_info.csv')
layout = np.load(f'{root}/stim_layout.npz', allow_pickle=True)

metamer_rsp = avr[:, layout['slice_metamer'][0]:layout['slice_metamer'][1]]
nsd_rsp = avr[:, layout['slice_nsd'][0]:layout['slice_nsd'][1]]

trials = np.load(f'{root}/{area}/trials_rsp.npz')
trials_rsp = trials['trials_rsp']                    # (n_cell, n_repeat, 2000)
n_repeat_valid = trials['n_repeat_valid']            # per-cell valid repeat count
```

---

## 与 Metamer1k 的差异

| 项目 | Metamer1k | Metamer_NSD |
|------|-----------|-------------|
| 脚本 | `Metamer1k_Summary.py` | `MetamerNSD_Summary.py` |
| `select_mod` | `'Metamer_1k'` | `'Metamer_NSD'` |
| 保存目录 | `Raw_Metamer_1k` | `Metamer_NSD_2k` |
| 脑区 | ML, MSB, AL, ASB | ML, MSB, AL, ASB, **ALO** |
| 图像数 `N_IMG` | 1000 | 2000（1000 metamer + 1000 NSD） |
| FOB 格式 | 多种（FOB72/STI150） | FOB72 |
| 热图文件名 | `heatmap_1k.png` | `heatmap_2k.png` |
| 刺激映射 | 无 | `stim_layout.npz` |
| 增量导出 | 无 | 有（按 site mtime） |

下游分析若只关心 metamer 或 NSD 子集，请用 `stim_layout.npz` 中的切片；导出列索引 0 起为 metamer。
