# Metamer Single Bubble 细胞筛选与导出 — 数据结构与使用说明

从 `Metamer_Singlebubble_v251107` 刺激集的 site class 中筛选细胞，导出四脑区（ML、MSB、AL、ASB）的标准化响应数据。流程与 `Metamer1k_Summary.py` / `MetamerNSD_Summary.py` 一致。

**导出脚本**：`MetamerBubble_Summary.py`

**默认保存根目录**：

`E:\#Preprocessed_Data\Selected_Cells\Metamers\Metamer_SingleBubble`

---

## 刺激集布局

`Metamer_Singlebubble_v251107` 刺激集共 **4540** 张图（见 `Py_Structure/Info_Files/Metamer_Singlebubble_v251107.tsv`）：

| 原始 stim 索引 | 数量 | 内容 | 是否导出 |
|----------------|------|------|----------|
| 0–299 | 300 | STI150 FOB（调谐计算） | 仅 FOB 导出 |
| 300–1299 | 1000 | Metamer shuffle 块 | 是（列 0–999） |
| 1300–1339 | 40 | Metamer 尾部 | 是（列 1000–1039） |
| 1340–2939 | 1600 | Bubble / Occluded 块 | 是（列 1040–2639） |
| 2940–4539 | 1600 | Rested 块 | 是（列 2640–4239） |

导出数组的**列索引**与原始 stim 索引关系：

```
export_col = raw_stim_index - 300
```

切片常量（与脚本内一致）：

```python
SLICE_METAMER      = slice(0, 1000)
SLICE_METAMER_TAIL = slice(1000, 1040)
SLICE_BUBBLE       = slice(1040, 2640)
SLICE_REST         = slice(2640, 4240)
```

Metamer shuffle 块（列 0–999）附加元信息（见 `stim_layout.npz`）：

- `metamer_img_index`：1–40，每图重复 25 次
- `metamer_shuffle_level`：0–4，每 level 40 张图

Bubble 空间 mask 见 `Py_Structure/Info_Files/Masks_Metamer_Singlebubble_v251107.npz`（未写入本导出，需时用 `Load_Info(..., load_mask=True)` 单独加载）。

---

## 目录结构

```
Metamer_SingleBubble/
├── README_Metamer_SingleBubble.md   # 本说明
├── stim_layout.npz                  # 刺激索引映射（四脑区共享）
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
    ├── heatmap_4240.png
    ├── heatmap_fob.png
    └── raster_first40.png
```

---

## 根目录共享文件

### `stim_layout.npz`

| 字段 | shape | 说明 |
|------|-------|------|
| `data_ids` | (4240,) | 原始 stim 索引（300–4539） |
| `stim_index` | (4240,) | 导出列索引（0–4239） |
| `stim_set` | (4240,) | TSV 中 `Stim_Set` 列 |
| `category` | (4240,) | TSV 中 `Category` 列 |
| `object_id` | (4240,) | TSV 中 `Object` 列 |
| `stim_type` | (4240,) | `'metamer'` / `'metamer_tail'` / `'bubble'` / `'rest'` |
| `metamer_img_index` | (4240,) | Metamer 块内原图编号 1–40；其余 NaN |
| `metamer_shuffle_level` | (4240,) | Metamer 块内 shuffle level 0–4；其余 NaN |
| `slice_metamer` | (2,) | `[0, 1000)` |
| `slice_metamer_tail` | (2,) | `[1000, 1040)` |
| `slice_bubble` | (2,) | `[1040, 2640)` |
| `slice_rest` | (2,) | `[2640, 4240)` |
| `n_img` / `n_metamer` / `n_bubble` / `n_rest` | scalar | 4240 / 1000 / 1600 / 1600 |

---

## 各脑区文件（`{area}/`）

### 响应矩阵

| 文件 | shape | 说明 |
|------|-------|------|
| `avr_rsp.npy` | `(n_cell, 4240)` | 50–219 ms 窗口内 spike count，trial 平均 |
| `psth.npy` | `(n_cell, 4240, 450)` | 全时段 PSTH，trial 平均；时间轴 -100–349 ms（1 ms bin，onset=300） |
| `trials_raw.npy` | `(n_cell, n_repeat_max, 4240, 450)` | trial 级原始 spike count；不足 repeat 处为 NaN |
| `trials_rsp.npz` | `trials_rsp`: `(n_cell, n_repeat_max, 4240)` | trial 级 50–219 ms 窗口响应 |
| `trials_raw_binned5ms.npz` | `trials`: `(n_cell, n_repeat_max, 4240, 90)` | 5 ms 分箱后的 trial 数据 |

### 细胞元信息

| 文件 | 说明 |
|------|------|
| `cell_site_info.csv` / `.joblib` | 每细胞一行：`global_idx`, `site_name`, `date`, `subject`, `local_cell_idx`, `stimset`, `dprime_face`, `dprime_body`, `ceiling_index`, `n_repeat` |
| `site_manifest.joblib` | 按 recording site 的轻量索引，供独立 FOB 重跑使用 |

### FOB 调谐

| 文件 | shape | 说明 |
|------|-------|------|
| `fob_avr.npy` | `(n_cell, 150)` | STI150 平均响应；有效列 150（由 300 FOB 刺激双次平均） |
| `fob_by_trial.npz` | `fob_by_trial`: `(n_cell, n_repeat_max, 150)` | trial 级 FOB |
| `fob_meta.npz` | `fob_valid_len`, `fob_style` | 每细胞有效 FOB 长度（150）与样式（`STI150`） |

### 质控图

| 文件 | 说明 |
|------|------|
| `heatmap_4240.png` | 4240 刺激 per-neuron z-score 热图 |
| `heatmap_fob.png` | STI150 FOB per-neuron z-score 热图 |
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

仅处理 `stimset == 'Metamer_Singlebubble_v251107'` 的 site class 文件。

---

## 运行方式

### 0. 建立 site-class 轻量索引（首次或新增 site 后）

```bash
cd _Projects/_Cell_Selection
python Site_Class_Lite_Scan.py
```

索引保存至：

`E:\#Preprocessed_Data\SiteClass\Metamers\site_class_lite_index.joblib`

### 1. 导出 Metamer Single Bubble 数据

在 IDE 中逐 cell 运行 `MetamerBubble_Summary.py`，或：

```bash
cd _Projects/_Cell_Selection
python MetamerBubble_Summary.py
```

### 开关

| 变量 | 默认 | 说明 |
|------|------|------|
| `RUN_LITE_SCAN` | `False` | `True` 时在 summary 内刷新索引 |
| `RUN_SITE_REFRESH` | `False` | 一次性刷新 site class（MF→ML、重算 ceiling/FOB） |
| `RUN_FOB_EXPORT` | `True` | 主流程后独立导出 FOB |

### 输入路径

| 路径 | 内容 |
|------|------|
| `E:\#Preprocessed_Data\SiteClass\Metamers\ML_MSB` | ML / MSB site class |
| `E:\#Preprocessed_Data\SiteClass\Metamers\AL_ASB` | AL / ASB site class |
| `Py_Structure/Info_Files/Metamer_Singlebubble_v251107.tsv` | 刺激元信息 |

---

## 读取示例

```python
import numpy as np
import pandas as pd

root = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Metamer_SingleBubble'
area = 'MSB'

avr = np.load(f'{root}/{area}/avr_rsp.npy')          # (n_cell, 4240)
info = pd.read_csv(f'{root}/{area}/cell_site_info.csv')
layout = np.load(f'{root}/stim_layout.npz', allow_pickle=True)

metamer_rsp = avr[:, layout['slice_metamer'][0]:layout['slice_metamer'][1]]
bubble_rsp = avr[:, layout['slice_bubble'][0]:layout['slice_bubble'][1]]
rest_rsp = avr[:, layout['slice_rest'][0]:layout['slice_rest'][1]]

trials = np.load(f'{root}/{area}/trials_rsp.npz')
trials_rsp = trials['trials_rsp']                    # (n_cell, n_repeat, 4240)
n_repeat_valid = trials['n_repeat_valid']            # per-cell valid repeat count
```

---

## 与其他 Metamer 导出的差异

| 项目 | Metamer1k | Metamer_NSD | Metamer Single Bubble |
|------|-----------|-------------|------------------------|
| 脚本 | `Metamer1k_Summary.py` | `MetamerNSD_Summary.py` | `MetamerBubble_Summary.py` |
| `select_mod` | `'Metamer_1k'` | `'Metamer_NSD'` | `'Metamer_Bubble'` |
| 保存目录 | `Raw_Metamer_1k` | `Metamer_NSD_2k` | `Metamer_SingleBubble` |
| 图像数 `N_IMG` | 1000 | 2144 | 4240 |
| FOB 样式 | 多种 | FOB72 | STI150 |
| 热图文件名 | `heatmap_1k.png` | `heatmap_2k.png` | `heatmap_4240.png` |
| 刺激映射 | 无 | `stim_layout.npz` | `stim_layout.npz` |

下游分析若只需 metamer / bubble / rest 子集，请用 `stim_layout.npz` 中的切片索引，勿假设列 0 起即为单一条件。
