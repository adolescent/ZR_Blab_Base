# 50D Object Space 分析 — 数据结构与使用说明

基于 Bao et al. (Nature IT, 2020) 的方法：AlexNet fc6 → NSD1k PCA 得到 50 维 object space，将 metamer 刺激嵌入该空间，拟合神经元偏好轴，并分析 metamer shuffle 效应。

默认保存根目录：

`E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis`

---

## 目录结构

```
Analysis/
├── README.md
├── nsd1k_obj_space_step1.npz
├── metamer1k_obj_space_step2.npz
├── shuffle_axis.npz
├── figures/                                    # 共享图（与脑区无关）
│   ├── Test_Obj_Space_Rsp/
│   ├── Obj_Space_Thought_Reversed/
│   └── …
├── MSB/
│   ├── obj_axis_fit.npz
│   ├── obj_axis_summary.csv
│   ├── shuffle_neuron.npz
│   ├── shuffle_neuron_summary.csv
│   ├── mediation.npz
│   └── figures/                                # 该脑区专属图
│       ├── Test_Obj_Space_Rsp/
│       ├── Obj_Space_Thought_Reversed/
│       └── Obj_Space_Shuffle_Intersected/
├── ML/  ASB/  AL/
│   └── …（同上）
```

**根目录 `figures/`**：object space、shuffle 轴等共享图。

**`{area}/figures/{脚本名}/`**：各脑区群体统计与 demo 细胞图（PNG，150 dpi）。

---

## 图片命名（按脚本）

### `Test_Obj_Space_Rsp`（共享 + 脑区）

| 文件名 | 说明 |
|--------|------|
| `step1_pc{N}_extremes.png` | NSD1k 在 PC N 上的极端图片 |
| `step1_pca_variance.png` | PCA 累积方差曲线 |
| `step2_metamer_pc12.png` | Metamer 在 PC1–PC2 上的分布 |
| `cell{id}_nsd_extremes.png` | 单细胞 NSD 轴极端图 |
| `cell{id}_metamer_tuning.png` | 单细胞 metamer 轴载荷–响应 |
| `pop_r2_hist.png` | 群体 R² 直方图 |
| `pop_axis_angle_matrix.png` | 神经元偏好轴两两夹角矩阵 |
| `demo_cell{id}_load_rsp.png` | Demo：1000 metamer 载荷 vs 响应 |
| `pop_ramp_heatmap.png` | Fig3c 风格 ramp tuning 热图 |

### `Obj_Space_Thought_Reversed`

| 文件名 | 说明 |
|--------|------|
| `shuffle_r2_ani_inani.png` | Shuffle 轴拟合 R²（Ani / Inani） |
| `shuffle_r2_all.png` | Shuffle 轴 R²（40 obj） |
| `shuffle_axis_pc12_*.png` | Shuffle 轴在 PC1–PC2 上的方向 |
| `shuffle_tuning_{ani,inani,all}.png` | 沿 shuffle 轴的载荷–level 关系 |
| `pop_shuffle_*.png` | 群体 shuffle 轴编码 / 夹角统计 |
| `demo_cell{id}_shuffle_*.png` | 单细胞 shuffle 载荷 demo |

### `Obj_Space_Shuffle_Intersected`

| 文件名 | 说明 |
|--------|------|
| `test1_slope_scatter.png` | slope_load vs slope_rsp |
| `test2_delta_r2.png` | 增量 ΔR² 检验 |
| `demo_cell{id}_panel_{A,B,C}.png` | 中介 demo 三联图 |
| `test6a_cor_coupling.png` | Ani / Inani load–rsp 耦合分布 |
| `demo_{ani,inani}_cell{id}_obj{n}.png` | 筛选后的单 object 轨迹 |

---

## 共享 npz 文件（根目录）

### `nsd1k_obj_space_step1.npz`

| 字段 | shape | 说明 |
|------|-------|------|
| `fc6` | (1000, 4096) | NSD1k AlexNet fc6 |
| `coords` | (1000, 50) | 50D 坐标 |
| `pc_mean` / `pc_components` | — | PCA 投影参数 |
| `cumvar` / `ev_ratio` | — | 解释方差 |
| `img_paths` | (1000,) | 图片路径 |

### `metamer1k_obj_space_step2.npz`

Metamer 0001–1000 的 fc6 与 50D 嵌入坐标。

### `shuffle_axis.npz`

`w_ani` / `w_inani` / `w_all`、`load_*`、`r2_*`：shuffle 敏感轴（脑区无关）。

---

## 各脑区 npz / csv（`Analysis/{area}/`）

| 文件 | 说明 |
|------|------|
| `obj_axis_fit.npz` | 神经元 50D 偏好轴、`meta_load`、R² |
| `obj_axis_summary.csv` | 每细胞 R² 与 NSD 极端图索引 |
| `shuffle_neuron.npz` | 偏好轴 vs shuffle 轴夹角与 R² |
| `shuffle_neuron_summary.csv` | 上表 CSV 版 |
| `mediation.npz` | 斜率相关、ΔR²、avg_load/rsp 等 |

字段细节见上文表格或脚本内注释。

---

## 脚本与运行

| 脚本 | 作用 |
|------|------|
| `All_Brain_Areas.py` | 批量：四脑区数据 + **全部标准图**（不弹窗） |
| `Test_Obj_Space_Rsp.py` | 交互分析；`SAVE_FIGURES=True` 时同步存图 |
| `Obj_Space_Thought_Reversed.py` | 同上 |
| `Obj_Space_Shuffle_Intersected.py` | 同上 |
| `obj_space_paths.py` | 路径常量 |
| `obj_space_plot.py` | `finish_fig()` 存图 helper |
| `obj_space_figures.py` | 批量出图（被 `All_Brain_Areas` 调用） |

### 批量（数据 + 图）

```bash
cd OBJ_Space_Test
python All_Brain_Areas.py
```

### 交互脚本

各脚本顶部：

```python
SAVE_FIGURES = True   # 是否写入 PNG
SHOW_FIGURES = True   # 是否 plt.show()
check_area = 'ASB'    # 脑区
```

改 `check_area` 后逐 cell 运行；图写入 `Analysis/{area}/figures/…` 或共享 `Analysis/figures/…`。

### 仅重出图（需已有 npz）

```python
from obj_space_figures import generate_all_figures
generate_all_figures(savepath, cell_rootpath, show=False)
```

---

## 输入数据

| 路径 | 内容 |
|------|------|
| `E:\#Stimsets\NSD1000\` | 1000 张 NSD 图 |
| `E:\#Stimsets\Metamer_P4_C4321_Object_STI150_1300\{0001..1000}.jpg` | metamer 刺激 |
| `…\Raw_Metamer_1k\{area}\avr_rsp.npy` | `(n_cell, 1000)` 平均响应 |
