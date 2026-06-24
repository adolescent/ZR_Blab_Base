# Metamer_NSD 细胞筛选与导出 — 数据结构与使用说明

从 `Metamer_NSD` 刺激集的 site class 中筛选细胞，导出四脑区（ML、MSB、AL、ASB）的标准化响应数据。流程与 `Metamer1k_Summary.py` 一致，仅刺激集与图像数量不同。

**导出脚本**：`MetamerNSD_Summary.py`

**默认保存根目录**：

`E:\#Preprocessed_Data\Selected_Cells\Metamers\Metamer_NSD_2k`

---

## 刺激集布局

`Metamer_NSD` 刺激集共 **2216** 张图（见 `Py_Structure/Info_Files/Metamer_NSD.tsv`）：

| 原始 stim 索引 | 数量 | 内容 | 是否导出 |
|----------------|------|------|----------|
| 0–71 | 72 | FOB72（调谐计算） | 仅 FOB 导出 |
| 72–215 | 144 | FOB 重复块 | 是（列 0–143） |
| 216–1215 | 1000 | Metamer | 是（列 144–1143） |
| 1216–2215 | 1000 | NSD1k | 是（列 1144–2143） |

导出数组的**列索引**与原始 stim 索引关系：

```
export_col = raw_stim_index - 72
```

切片常量（与脚本内一致）：

```python
SLICE_FOB_REPEAT = slice(0, 144)
SLICE_METAMER    = slice(144, 1144)    # 1000 metamer
SLICE_NSD        = slice(1144, 2144)   # 1000 NSD
```

---

## 目录结构

```
Metamer_NSD_2k/
├── README_Metamer_NSD.md          # 本说明（可选复制到数据目录）
├── stim_layout.npz                # 刺激索引映射（四脑区共享）
├── ML/
├── MSB/
├── AL/
└── ASB/
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
| `data_ids` | (2144,) | 原始 stim 索引（72–2215） |
| `stim_index` | (2144,) | 导出列索引（0–2143） |
| `stim_set` | (2144,) | TSV 中 `Stim_Set` 列 |
| `category` | (2144,) | TSV 中 `Category` 列 |
| `object_id` | (2144,) | TSV 中 `Object` 列（metamer object id；FOB/NSD 为 -1 或 NSD id） |
| `stim_type` | (2144,) | `'fob_repeat'` / `'metamer'` / `'nsd'` |
| `slice_fob_repeat` | (2,) | `[0, 144)` |
| `slice_metamer` | (2,) | `[144, 1144)` |
| `slice_nsd` | (2,) | `[1144, 2144)` |
| `n_img` / `n_metamer` / `n_nsd` | scalar | 2144 / 1000 / 1000 |

---

## 各脑区文件（`{area}/`）

### 响应矩阵

| 文件 | shape | 说明 |
|------|-------|------|
| `avr_rsp.npy` | `(n_cell, 2144)` | 50–219 ms 窗口内 spike count，trial 平均 |
| `psth.npy` | `(n_cell, 2144, 450)` | 全时段 PSTH，trial 平均；时间轴 -100–349 ms（1 ms bin，onset=300） |
| `trials_raw.npy` | `(n_cell, n_repeat_max, 2144, 450)` | trial 级原始 spike count；不足 repeat 处为 NaN |
| `trials_rsp.npz` | `trials_rsp`: `(n_cell, n_repeat_max, 2144)` | trial 级 50–219 ms 窗口响应 |
| `trials_raw_binned5ms.npz` | `trials`: `(n_cell, n_repeat_max, 2144, 90)` | 5 ms 分箱后的 trial 数据 |

### 细胞元信息

| 文件 | 说明 |
|------|------|
| `cell_site_info.csv` / `.joblib` | 每细胞一行：`global_idx`, `site_name`, `date`, `subject`, `local_cell_idx`, `stimset`, `dprime_face`, `dprime_body`, `ceiling_index`, `n_repeat` |
| `site_manifest.joblib` | 按 recording site 的轻量索引，供独立 FOB 重跑使用 |

### FOB 调谐

| 文件 | shape | 说明 |
|------|-------|------|
| `fob_avr.npy` | `(n_cell, 150)` | FOB72 平均响应；有效列 72，其余 NaN padding |
| `fob_by_trial.npz` | `fob_by_trial`: `(n_cell, n_repeat_max, 150)` | trial 级 FOB |
| `fob_meta.npz` | `fob_valid_len`, `fob_style` | 每细胞有效 FOB 长度（72）与样式（`FOB72`） |

### 质控图

| 文件 | 说明 |
|------|------|
| `heatmap_2k.png` | 2144 刺激 per-neuron z-score 热图 |
| `heatmap_fob.png` | FOB72 per-neuron z-score 热图 |
| `raster_first40.png` | 前 40 张刺激的平均 PSTH raster（5 ms bin） |

---

## 筛选参数

与 `Metamer1k_Summary.py` 相同：

| 参数 | 值 | 说明 |
|------|-----|------|
| `CEILING_THRES` | 0.3 | noise ceiling 阈值 |
| `DP_THRES` | 0.5 | FOB D' 阈值 |
| `AREA_PREFER` | ML/AL→Face, MSB/ASB→Body | 脑区偏好类别 |
| `TIME_SLICE` | 150:320 | 响应窗口（50–219 ms） |
| `MAX_REPEAT` | 20 | trial 维上限 |

仅处理 `stimset == 'Metamer_NSD'` 的 site class 文件。

---

## 运行方式

### 0. 建立 site-class 轻量索引（首次或新增 site 后）

Summary 不再遍历文件夹内全部 joblib，而是先读索引，只加载匹配 `select_mod` 的 site。

```bash
cd _Projects/_Cell_Selection
python Site_Class_Lite_Scan.py
```

索引保存至：

`E:\#Preprocessed_Data\SiteClass\Metamers\site_class_lite_index.joblib`（及同名 `.csv`）

扫描时**优先从文件名**解析 `{site}_{areas}_{stimset}.joblib`，无需反序列化 SRS；仅当文件名无法匹配已知 stimset 时才 fallback 到 `joblib.load`。

增量更新：再次运行 scan 时，仅重新解析 mtime/size 变化的文件。

### 1. 导出 Metamer_NSD 数据

在 IDE 中逐 cell 运行 `MetamerNSD_Summary.py`，或：

```bash
cd _Projects/_Cell_Selection
python MetamerNSD_Summary.py
```

### 开关

| 变量 | 默认 | 说明 |
|------|------|------|
| `RUN_LITE_SCAN` | `False` | `True` 时在 summary 内刷新索引（建议单独跑 `Site_Class_Lite_Scan.py`） |
| `RUN_SITE_REFRESH` | `False` | 一次性刷新 site class（MF→ML、重算 ceiling/FOB） |
| `RUN_FOB_EXPORT` | `True` | 主流程后独立导出 FOB |

### 输入路径

| 路径 | 内容 |
|------|------|
| `E:\#Preprocessed_Data\SiteClass\Metamers\ML_MSB` | ML / MSB site class |
| `E:\#Preprocessed_Data\SiteClass\Metamers\AL_ASB` | AL / ASB site class |
| `Py_Structure/Info_Files/Metamer_NSD.tsv` | 刺激元信息 |

---

## 读取示例

```python
import numpy as np
import pandas as pd
import joblib as jl

root = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Metamer_NSD_2k'
area = 'MSB'

avr = np.load(f'{root}/{area}/avr_rsp.npy')          # (n_cell, 2144)
info = pd.read_csv(f'{root}/{area}/cell_site_info.csv')
layout = np.load(f'{root}/stim_layout.npz', allow_pickle=True)

metamer_rsp = avr[:, layout['slice_metamer'][0]:layout['slice_metamer'][1]]
nsd_rsp = avr[:, layout['slice_nsd'][0]:layout['slice_nsd'][1]]

trials = np.load(f'{root}/{area}/trials_rsp.npz')
trials_rsp = trials['trials_rsp']                    # (n_cell, n_repeat, 2144)
n_repeat_valid = trials['n_repeat_valid']            # per-cell valid repeat count
```

---

## 与 Metamer1k 的差异

| 项目 | Metamer1k | Metamer_NSD |
|------|-----------|-------------|
| 脚本 | `Metamer1k_Summary.py` | `MetamerNSD_Summary.py` |
| `select_mod` | `'Metamer_1k'` | `'Metamer_NSD'` |
| 保存目录 | `Raw_Metamer_1k` | `Metamer_NSD_2k` |
| 图像数 `N_IMG` | 1000 | 2144（含 144 FOB 重复 + 1000 metamer + 1000 NSD） |
| FOB 样式 | 多种（FOB72/STI150） | FOB72 |
| 热图文件名 | `heatmap_1k.png` | `heatmap_2k.png` |
| 刺激映射 | 无 | `stim_layout.npz` |

下游分析若只需 metamer 或 NSD 子集，请用 `stim_layout.npz` 中的切片索引，勿假设列 0 起即为 metamer。
