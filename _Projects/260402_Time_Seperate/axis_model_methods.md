# Article methodology and data processing questions
_Exported on 4/20/2026 at 18:01:59 GMT+8 from Cursor (3.1.15)_

---

**User**

https://www.nature.com/articles/s41586-026-10267-3#Sec7 关于这篇文章中使用的细节方法，我对数据处理的方式有以下几个疑问，请为我回答：
1.文中的normalize是怎么做的，记录到一个神经元50-200ms对所有图片的响应强度后，如何做归一化？50-200ms的响应是怎么算的，平均还是求和？
2.根据总体响应做的分析如何在后续使用20ms的时间窗使用？这个窗口是硬切割的还是滑动的？如何把20ms的响应与150ms的响应类比

---

**Cursor**

根据论文 Methods 中 "Average response profile"、"Preferred axis of cells" 和 "Time-varying axis analysis" 几节（文末 markdown 文件的第 290–344 行），我来逐一回答您的问题。首先澄清一下：文中用的窗口是 **50–220 ms**（不是 50–200 ms），这是 stimulus onset 之后的时间。

---

## 问题 1：50–220 ms 响应怎么算？normalize 如何做？

### (a) 50–220 ms 的响应 = 平均，不是求和

Methods 原文（第 294 行）：

> "**Mean responses** of each cell to each stimulus were computed in a 50–220 ms window after stimulus onset."

也就是对该 170 ms 窗口内的 spike 取**平均发放率**（mean firing rate，单位 spikes/s），而不是 spike count 求和。这也与第 290 行 FSI 公式里写的 "_r_ is the **average** neuronal response in a 50–220 ms window" 以及第 312 行 axis 回归公式里 **r** 是 "firing-rate response" 一致。

实操上对每个细胞每张图：
- 先取每个 trial 在 [50, 220] ms 内的 spike 数 / 0.17 s → 该 trial 的 firing rate
- 再对重复 trial 取平均 → 得到该细胞对该图的 mean response `r_i`

### (b) Normalize 的做法 = per-cell min–max 到 [0, 1]

同一段（第 294 行）紧接着写：

> "Responses were then normalized for each cell to the range [0, 1], where the minimum response was assigned 0 and maximum was assigned 1."

即对每个神经元，**在所有图片（1,525 张脸 + 1,392 张物体）上**：

\[
\tilde{r}_i = \frac{r_i - \min_j r_j}{\max_j r_j - \min_j r_j}
\]

这样做纯粹是为了**可视化时把不同发放率量级的细胞放在一起**（如 Fig. 1g,h、Ext. Data Fig. 1e 的 response profile heatmap）。

### (c) 需要区分"两种 normalize"

这里容易混淆，论文里实际上有两处不同的 normalize：

| 用途 | 对象 | 方法 |
|---|---|---|
| **Response profile 可视化** | 每个细胞的响应 `r_i` | min–max 到 [0, 1] |
| **Axis 回归 / 特征空间** | 60 维特征空间的每个维度 | z-score（mean=0, sd=1），见第 300 行 |

⚠️ 关键点：**拟合 axis（Fig. 2、3、4）时用的是原始 firing rate（或减去均值），不是 [0,1] 归一化后的值**。第 312 行的回归公式里 **r** 明确写的是 "firing-rate response"，并且做了 `r − r̄` 中心化。而 Fig. 5 single-stimulus axis-change score 甚至特别强调 "**avoided any per-cell normalization and worked with raw firing rates**"（第 348 行）。所以 [0,1] 这个归一化**只用在展示平均 response profile 的图里**，不要把它带进 axis/decoding 分析。

---

## 问题 2：20 ms 窗口怎么用？如何与 170 ms 窗口的"总体响应"比较？

### (a) 20 ms 是滑动窗口（sliding），不是硬切

Methods "Time-varying axis analysis"（第 332 行）：

> "We fitted axes to the average neural responses in **20-ms windows** over the trial duration of 0–300 ms after stimulus onset."

配合正文描述（第 122 行）和 FSI 计算（第 280 行）：

> "we computed face and object axes using a **20-ms sliding window**"
> "peak _d_' over a **20-ms sliding window** between 80 ms and 140 ms"

所以是**滑动窗口**，步长论文没有在正文明说（按常规和 d' 能做 peak search 的写法，步长通常是 1 ms，每 1 ms 取一次 [T, T+20] 的平均响应，得到一条随 latency 变化的曲线）。

### (b) 每个 20 ms 窗口做的操作与 50–220 ms 完全同构

对任一 latency `T`：
1. 对每个细胞每张图，计算 [T, T+20] ms 内的 mean firing rate `r_i(T)`
2. 用这 `r_i(T)` 作为回归目标，对 60 维 AlexNet 特征做 **和 50–220 ms 一模一样的线性回归**（第 310 行公式）
3. 得到该细胞在时间 T 的 "time-varying face axis" 和 "time-varying object axis"，每条都是 60 维向量

注意：**虽然两个窗口长度不同（20 ms vs 170 ms），但得到的 axis 都是同一个 60 维特征空间里的方向向量**，所以它们的量纲是可比的——axis 的朝向不依赖于窗口内 firing rate 的绝对大小。

### (c) 怎么把 20 ms 的 axis 和 170 ms 的 axis "类比"

论文**不是**直接比较"20 ms 的响应强度 vs 170 ms 的响应强度"，而是**比较两个窗口各自回归出来的 axis 之间的夹角**（cosine similarity）。见第 332 行：

> "To quantify the alignment between axes at different latencies, we computed **cosine similarity** between the time-varying face or object axis at each latency and the trial-wide object axis computed across 50–220 ms"

也就是说：

- "总体 axis" `v_overall` = 用 50–220 ms 响应回归出的 60 维方向向量（Fig. 2）
- "时变 axis" `v(T)` = 用 [T, T+20] ms 响应回归出的 60 维方向向量
- 比较量：`cos(v(T), v_overall)` 随 T 变化（Fig. 3e 的每一行就是一个细胞的这条曲线）

这样做绕开了"短窗口 spike 少、长窗口 spike 多"的问题——**因为 axis 是归一化的方向，而不是响应幅度**。回归里的 `(F^T F)^{-1}` 也让 axis 的尺度只依赖于特征矩阵 F，不依赖响应的绝对量级。

### (d) 对您实现 A21_PCA_Cross_Decode.py 的建议

基于以上，您在实现 cross-decode / cross-axis 分析时：

1. 20 ms 窗口每一步的"响应"用**平均 firing rate**（spike count / 0.02 s），与大窗口保持同一量纲；
2. **不要**对每个细胞做 [0,1] min–max 再送去回归——那是 profile 图专用；如果做了会让不同细胞的 axis 权重被其 dynamic range 反向缩放；
3. 回归前对响应做 `r − r̄`（mean-center），和 Methods 第 310 行一致；
4. 比较不同窗口之间的 "编码"时，标准做法是比较 axis 方向（cosine similarity 或 R² cross-prediction），而不是直接比响应值；
5. 滑动窗口步长一般取 1 ms 或 5 ms；如果只想看几个代表性 latency，论文在 Fig. 4 就直接用了三个硬切窗口 **80–100 ms / 120–140 ms / 160–180 ms**（见第 156 行），这是可以接受的折中。

如果您希望我把上面第 2–4 点对照到当前 `A21_PCA_Cross_Decode.py` 里检查一下实现是否符合论文，可以告诉我。

---

**User**

好的，这篇文章是如何定义object轴和face轴的？

---

**Cursor**

根据 Methods 的 "AlexNet general-object space and face space"（第 296–304 行）和 "Preferred axis of cells"（第 306–312 行），以及正文 Fig. 1 介绍（第 90、102 行），"轴（axis）"的定义有两层——**特征空间的轴（feature axis）** 和 **每个细胞的偏好轴（preferred axis）**。这两件事要分开看。

---

## 一、先建 60 维特征空间（所有细胞共享的坐标系）

论文用 AlexNet 建了**两个独立的 60 维空间**，它们只是用来表达每张图的特征向量 **f**，与神经活动无关：

### (a) 60-D "general-object space"（主要用于 Fig. 2–3）

流程：

1. 把一批刺激图（**100 张脸 + 1,292 个物体**）送进 pretrained AlexNet
2. 取 **fc6 层的 4,096 维特征**
3. 对这 `(100+1,292) × 4,096` 的矩阵做 **PCA**，取前 60 个 PC
4. 对每个 PC 维度再做 z-score：让所有刺激投影到该维后 mean=0, sd=1

这个空间对 objects 解释了 80.9%、对 faces 解释了 61.6% 的 fc6 方差。

### (b) 60-D "face space"（主要用于 Fig. 4）

同样流程，但 PCA 只用 **1,425 张脸**，所以它强调 face identity 之间的差异。

⚠️ 注意：这两个都叫"60-D 特征空间"，但**"object 轴 / face 轴"不是这两个空间的 PC 方向**。PC 只是给你一组坐标基，用来写每张图的 60 维 feature 向量 **f**。

---

## 二、每个 cell 的 "object axis" 和 "face axis" = 线性回归权重

真正意义上的"轴"是**每个细胞独立拟合出来的一个 60 维权重向量** c，使得
\[
r = \mathbf{c}\cdot\mathbf{f}
\]
即该细胞对一张图的响应近似等于该图的特征向量 **f** 沿 **c** 方向的投影长度。这个 **c** 就是"preferred axis"。

Methods 第 310 行给出的闭式解（去均值 + 普通最小二乘）：

\[
P_{\text{lin}} = (\mathbf{r}-\bar r)\, F\, (F^T F)^{-1}
\]

- **r** ：该 cell 对 n 张图的 firing-rate 响应（1×n 向量，50–220 ms 平均）
- **F** ：n×60，每行是一张图在 60 维特征空间里的向量 **f**
- **P_lin** ：1×60，就是 axis 的权重向量 **c**

### 关键点：Object axis 和 Face axis 是用**不同的刺激子集**分别回归出来的

文中第 102 行（正文）写得最清楚：

> "For each face cell, we used responses to the **non-face object stimuli** (averaged over 50–220 ms) to compute a preferred axis, **the object axis**, by linearly regressing responses of the cell to the 60-dimension feature vectors corresponding to different objects. Similarly, we used responses to **face stimuli** to compute **a preferred face axis**."

所以对同一个细胞：

| 轴 | 用什么响应 r | 用什么特征矩阵 F | 得到的 c 含义 |
|---|---|---|---|
| **Object axis** | 该 cell 对 1,292 物体的 mean firing rate | 1,292 × 60（物体的 fc6-PCA 特征） | 该 cell 对物体**怎么调谐**的 60 维权重 |
| **Face axis** | 该 cell 对 1,425 张脸的 mean firing rate | 1,425 × 60（脸的 fc6-PCA 特征） | 该 cell 对脸**怎么调谐**的 60 维权重 |

⭐ **正因为两条轴来自不相交的刺激子集（脸 / 物体），它们才可能指向不同方向**——这也是整篇论文的立论基础：如果一个细胞真的是"单轴 domain-general"模型（如 AlexNet fc6 unit），这两条轴应该几乎一致；如果观察到它们不一致甚至反向（Fig. 2a、Fig. 2d 那个神经元），说明 face cell 用了两条独立的轴。

### Train / Test 划分

对每个类别（脸或物体），图片被预先划分为：
- **训练集**：1,292 个物体中的 1,192 / 1,425 张脸中的 1,325（正文第 222 行：每个 block 先放 training set 1 次重复，然后 test set 3 次重复）
- **测试集**：留出 100 个物体 / 100 张脸

回归用训练集做，**所有图里报告的 R² 都是 held-out test set 上的**（第 312 行）。入选 Fig. 2–4 分析的 cell 必须满足 object axis 和 face axis 在 test set 上 **R² > 0**（第 272 行）。

---

## 三、Time-varying axis（Fig. 3、4）的轴定义一模一样

只是把第二步里"50–220 ms 的 mean firing rate" 换成 **[T, T+20] ms 窗口的 mean firing rate**，其它（F、公式、train/test split）完全不变。所以每个细胞在每个 latency T 各有一条 face axis `v_face(T)` 和一条 object axis `v_obj(T)`，都是 60 维向量。

---

## 四、用作验证的 shuffle control（Extended Data Fig. 4a,b）

为了证明拟合出来的 axis 是真信号而不是过拟合：把 r 和 f 的对应关系**随机打乱**再跑一次同样的回归，用它的 test-R² 作为 null 分布。真实 axis 的 R² 明显高于这个 null——这就验证了"face axis 解释 face 响应、object axis 解释 object 响应"的显著性。

---

## 五、对应到 `A21_PCA_Cross_Decode.py` 的实现要点

如果要严格复刻论文定义，核心步骤应是：

```python
from sklearn.linear_model import LinearRegression

# F_face: (n_face, 60), F_obj: (n_obj, 60)  —— 已经 z-scored 的 fc6-PCA 特征
# r_face: (n_cells, n_face), r_obj: (n_cells, n_obj)  —— 50-220ms 平均 firing rate

face_axes = np.zeros((n_cells, 60))
obj_axes  = np.zeros((n_cells, 60))
for i in range(n_cells):
    face_axes[i] = LinearRegression().fit(F_face_train, r_face[i, train_idx]).coef_
    obj_axes[i]  = LinearRegression().fit(F_obj_train,  r_obj[i,  train_idx]).coef_

r2_face_test = ...  # 用 face_axes[i] 预测 r_face[i, test_idx]
r2_obj_test  = ...  # 同上
keep = (r2_face_test > 0) & (r2_obj_test > 0)   # 入选 cell
```

要注意的几点：

1. **F 和 r 都要先做 `- mean`（或 `LinearRegression(fit_intercept=True)`）**，否则不符合公式 (**r** − r̄)F(FᵀF)⁻¹。
2. **F_face_train 和 F_obj_train 是两套完全独立的特征矩阵**，不要混用成一个大矩阵一起回归——那就不是"face axis vs object axis"了。
3. **axis 应当建立在"脸图片集内部"和"物体图片集内部"**。Cross-prediction（Fig. 2c）才是把其中一个类别的 axis 拿去预测另一个类别的响应。
4. 同维度才能做 cosine similarity：face space 里的轴不能直接和 general-object space 里的轴比。Fig. 3 全程在 general-object space；Fig. 4 全程在 face space。
5. 做 time-varying axis 时，train/test 的图片划分要和 overall axis 保持一致，以便在同一套 test 图上算 R²。

如果您希望我直接对照一下当前 `A21_PCA_Cross_Decode.py` 和 `A3_Cross_Decode.py` 是否正确实现了 axis 和 cross-decode，可以让我读文件后给出具体修改建议。

---

**User**

从图3e来看，我认为作者使用了相同的空间来构建。请检查回应

---

**Cursor**

您完全正确，我之前的表述有歧义，现在更正并说清楚。

## 您的观察是对的：Fig. 3e 用的是**同一个**特征空间

Fig. 3e 的纵轴是 "cosine similarity between each cell's overall object axis and its time-varying face / object axes"。**Cosine similarity 要求两个向量在同一个坐标系里**，否则数学上没有意义。所以：

- overall object axis（50–220 ms）
- time-varying face axis（每个 20 ms 窗口）
- time-varying object axis（每个 20 ms 窗口）

**这三条轴都是在同一个 60 维 general-object space 里的向量**。这也正是论文在 Methods 里（第 300 行）特意说：

> "This general-object space was used for the analyses shown in **Figs. 2 and 3** and Extended Data Figs. 2a–j, 3a–j, q–v, 4, 5a–e."

整个 Fig. 3 都在这一个 60-D 空间里。

## 我上一条回复里哪里写得容易误解

我在表格里写：

| Object axis | F_obj: **1,292 × 60** |
| Face axis  | F_face: **1,425 × 60** |

这里 "1,292" 和 "1,425" 只是**行数**（参与回归的图片数量不同），而**列（那 60 个维度）是完全相同的那组 PC 基**——就是从 "100 张脸 + 1,292 个物体" 一起做 PCA 得到的 60 个方向。换句话说：

- **坐标系（60-D basis）：脸和物体共享一套**，由 general-object space 的 PCA 给定
- **回归时用的图片子集（F 的行）：脸轴只用脸图、物体轴只用物体图**
- **回归的响应 r：对应图片子集的 firing rate**

得到的两条 axis `c_face`、`c_obj` 都是**同一个 60-D 空间里的方向向量**，所以它们之间（以及与 overall axis 之间）可以直接做 `cos(c_a, c_b) = c_a·c_b / (‖c_a‖‖c_b‖)`。

## 需要注意的一个例外：Fig. 4 换了空间

Fig. 4 用的是**另一个** 60-D 空间——"face space"（PCA 只用 1,425 张脸，见 Methods 第 302 行）。在 Fig. 4 内部，face axis 和 object axis 又都在这**同一个 face space** 里，所以依然可以比较。

但是 **Fig. 3 的轴 ↔ Fig. 4 的轴 不能直接比 cosine**，因为它们是两个不同 PCA 得到的不同基。论文也从没跨空间比较过。

## 对实现的修正

如果您在 `A21_PCA_Cross_Decode.py` 里实现 axis 拟合，正确的做法应是：

```python
fc6_face = alexnet_fc6(face_images)   # (1425, 4096)
fc6_obj  = alexnet_fc6(obj_images)    # (1292, 4096)

pca = PCA(n_components=60).fit(np.vstack([fc6_face_subset, fc6_obj]))

F_face = pca.transform(fc6_face)   # (1425, 60)  同一组 PC
F_obj  = pca.transform(fc6_obj)    # (1292, 60)  同一组 PC
F_face = (F_face - F_all.mean(0)) / F_all.std(0)   # 每个维度 z-score
F_obj  = (F_obj  - F_all.mean(0)) / F_all.std(0)

for i in range(n_cells):
    obj_axis[i]  = LinearRegression().fit(F_obj_train,  r_obj[i, train_o]).coef_
    face_axis[i] = LinearRegression().fit(F_face_train, r_face[i, train_f]).coef_

sim = 1 - cosine(face_axis[i], obj_axis[i])
```

关键是 `PCA` **只拟合一次**（或用 Fig. 2 里那种 "100 faces + 1,292 objects" 的组合样本拟合），然后用同一个 `pca` 去 transform 脸和物体两组图。**不要对脸单独做一次 PCA、对物体单独做一次 PCA**——那就会得到两组不同的基，cosine 失去意义，也就无法复现 Fig. 3e。

感谢您指出这个问题，之前的表格表述确实没把"坐标系共享、图片子集不同"这个要点讲清楚。

---

**User**

face axis和object axis是空间中的两根轴，这两根轴是怎么做的？直接对空间中的face图左平均，所有神经元都一样吗，还是每个神经元都有自己的face和object轴？请为我详细说明这个问题。我想要知道3e图绘制的全部细节。

---

**Cursor**

这是个非常关键的问题，因为论文里"轴（axis）"这个词其实暗藏了一个很容易误解的技术细节。我来把 Fig. 3e 的构造从最底层一步一步拆开讲。

---

## 一、先纠正一个常见误解：轴 **不是** "在特征空间中把 face 图平均"

直觉上会想：既然已经有 60-D 特征空间，那"face 轴"是不是就是把所有 face 图的特征向量加起来取个平均方向？答案是**否**。

如果是那样做，**所有细胞的 face 轴都会是同一条**（就是脸图的质心方向），这就不存在"每个细胞对脸的调谐方向不同"这回事，论文整篇立论也就不成立。

论文定义的 axis 是：**每个神经元自己的线性回归权重**——即"沿着哪个方向前进，**这个细胞的** firing rate 增长最快"。每个 cell 有自己专属的一条 face axis 和一条 object axis。

---

## 二、单个细胞的 axis 严格定义（来自 Methods 第 306–312 行）

对**一个细胞 i**：

### 输入
- 响应向量 **r**_i = [r_{i,1}, r_{i,2}, ..., r_{i,n}] ∈ ℝ^{1×n}  
  每个 r_{i,j} 是该细胞对第 j 张图的**平均 firing rate**（在指定时间窗内计算）
- 特征矩阵 **F** ∈ ℝ^{n×60}  
  第 j 行 **f**_j 是第 j 张图在 60-D general-object space 里的特征向量

### 模型
\[
r_{i,j} \approx \mathbf{c}_i \cdot \mathbf{f}_j
\]
即假设这个细胞对图 j 的响应 ≈ 图 j 的特征沿某个 60-D 方向 **c**_i 的投影长度。

### 用最小二乘法拟合出 **c**_i
Methods 里给的闭式解（去均值 + OLS）：

\[
\mathbf{c}_i = (\mathbf{r}_i - \bar{r}_i)\,F\,(F^T F)^{-1} \in \mathbb{R}^{1\times 60}
\]

这条 **c**_i 就叫作**该细胞的 preferred axis**。

### 为什么每个细胞都不一样？
因为公式里的 **r**_i 是该细胞的响应。细胞 A 可能对"戴眼镜的脸"反应强（axis 指向 60-D 空间中"戴眼镜"方向），细胞 B 可能对"亚洲女性"反应强（axis 指向另一个方向）。F 是所有 cell 共享的（图片特征是客观的），但 **r**_i 是细胞特定的，所以回归出来的 **c**_i 也是细胞特定的。

### Face axis vs Object axis（对同一个细胞）
对同一个细胞 i，分别用两套 {r, F} 做两次回归：

| 轴 | **r**_i（这个细胞对哪些图的响应） | F（对应图的 60-D 特征） |
|---|---|---|
| **c**_i^{obj} object axis | 这个细胞对 **1,292 个物体** 的 mean firing rate | F_obj：1,292 × 60（在 general-object space 中） |
| **c**_i^{face} face axis | 这个细胞对 **1,425 张脸** 的 mean firing rate | F_face：1,425 × 60（同一个 general-object space） |

关键：两次回归用的是**同一个 60-D 基**（第 300 行说 general-object space 在 Figs. 2、3 全程一致），只是**参与回归的图片行不同**。这样 **c**_i^{face} 和 **c**_i^{obj} 才能直接算 cosine（Fig. 2、Fig. 3 的所有比较的基础）。

---

## 三、把 axis 拓展到时变情形（Fig. 3 专用）

之前的 axis 是用 50–220 ms 的 mean response 拟出来的，只有一条。要得到"随时间变化的 axis"，做法是：

**Methods 第 332 行：把 r 换成 20 ms 滑动窗口的平均 firing rate**，其他完全不变。

对每个细胞 i、每个 latency T（例如 T = 0, 1, 2, ..., 280 ms，步长 1 ms 或 5 ms）：

1. r_{i,j}^{face}(T) = 细胞 i 对第 j 张脸 在 [T, T+20] ms 窗口内的 mean firing rate（所有 trial 平均）
2. r_{i,j}^{obj}(T)  = 同上但对物体
3. 用 (r_i^{face}(T), F_face) 回归 → **c**_i^{face}(T)：该细胞在 T 时刻的 time-varying face axis
4. 用 (r_i^{obj}(T),  F_obj ) 回归 → **c**_i^{obj}(T)：该细胞在 T 时刻的 time-varying object axis

所以每个 cell 得到**两条 60-D 向量的时间序列**：
- {**c**_i^{face}(T)}_T
- {**c**_i^{obj}(T)}_T

---

## 四、Fig. 3e 的全部细节

### (a) 图的结构

- 两张热图（top / bottom），长得很像两张 raster
- **每一行 = 一个 cell**（共 n = 151 个 cell，ML face patch of monkey A，且要求 face axis、object axis 在 test set 上都 R² > 0）
- **横轴 = latency T**（0–300 ms）
- **颜色 = 一个 cosine similarity 值**（接近 +1 红，接近 −1 蓝，0 白）
- cell 从上到下按 **face selectivity d'** 排序（图左边的竖条就是每个 cell 的 d'）

### (b) 颜色到底在画什么

对每个细胞 i，先准备一条**参考轴**（整段 trial 的 object axis，称为 overall object axis）：

\[
\mathbf{c}_i^{\text{obj,overall}} = \text{regress}\big(\text{firing rate in 50–220 ms to objects}, \ F_{\text{obj}}\big)
\]

这条轴**不随时间变化**，就是 Fig. 2 里用的那条 object axis。

然后对每个 T 计算：

- **top panel**（face axis vs overall object axis）：
  \[
  S^{\text{face}}_i(T) = \frac{\mathbf{c}_i^{\text{obj,overall}} \cdot \mathbf{c}_i^{\text{face}}(T)}{\|\mathbf{c}_i^{\text{obj,overall}}\|\,\|\mathbf{c}_i^{\text{face}}(T)\|}
  \]

- **bottom panel**（object axis vs overall object axis）：
  \[
  S^{\text{obj}}_i(T) = \frac{\mathbf{c}_i^{\text{obj,overall}} \cdot \mathbf{c}_i^{\text{obj}}(T)}{\|\mathbf{c}_i^{\text{obj,overall}}\|\,\|\mathbf{c}_i^{\text{obj}}(T)\|}
  \]

把 S^{face}_i(T) 填进 top panel 第 i 行第 T 列的像素，S^{obj}_i(T) 填 bottom panel，就得到了 Fig. 3e 那两张热图。

### (c) 这张图想说什么

- **Bottom panel 几乎整行都红**：说明每个细胞在任何时刻的 object axis 都和它自己的 overall object axis 方向一致——object 的编码方向**稳定**。
- **Top panel 左边红、100 ms 之后变蓝（尤其是 d' 高的 cell）**：说明 face axis **早期** 和 object axis 对齐（红），**100 ms 之后翻转** 到反方向（蓝）。这就是论文核心结论 "rapid concerted axis change"。
- **按 d' 排序**让读者能看出这个"早红晚蓝"模式在强 face-selective cell 身上最显著。

### (d) 为什么参考的是"overall object axis"，不是"overall face axis"？

因为 object axis 是整个 trial 都稳定的（bottom panel 证明了这件事），所以它是一个**可靠的参考方向**（reliable reference direction）。用它做参考可以干净地看出 face axis 什么时候"跟它一致"、什么时候"背叛"它。如果拿 overall face axis 当参考，因为 face axis 本身在变，就会得到一个混乱的结果。

---

## 五、伪代码总览

```python
# --- 特征空间（对所有 cell 共用）---
F_face, F_obj = alexnet_fc6_pca60(...)   # 同一个 general-object space

# --- 每个 cell 的 overall object axis (50–220 ms) ---
c_obj_overall = np.zeros((n_cells, 60))
for i in range(n_cells):
    r = firing_rate[i, obj_stim_idx, window=(50, 220)].mean(trial_axis)  # (1292,)
    c_obj_overall[i] = LinearRegression(fit_intercept=True).fit(F_obj, r).coef_

# --- time-varying axes ---
latencies = np.arange(0, 281, 1)
S_face = np.zeros((n_cells, len(latencies)))
S_obj  = np.zeros((n_cells, len(latencies)))
for k, T in enumerate(latencies):
    for i in range(n_cells):
        r_f = firing_rate[i, face_stim_idx, window=(T, T+20)].mean(trial_axis)
        r_o = firing_rate[i, obj_stim_idx,  window=(T, T+20)].mean(trial_axis)
        c_face_T = LinearRegression().fit(F_face, r_f).coef_
        c_obj_T  = LinearRegression().fit(F_obj,  r_o).coef_
        S_face[i, k] = 1 - cosine(c_obj_overall[i], c_face_T)
        S_obj[i, k]  = 1 - cosine(c_obj_overall[i], c_obj_T)

sort_idx = np.argsort(-d_prime)   # d' 从高到低
plt.imshow(S_face[sort_idx], cmap='RdBu_r', vmin=-1, vmax=1)  # top
plt.imshow(S_obj[sort_idx],  cmap='RdBu_r', vmin=-1, vmax=1)  # bottom
```

---

## 六、三句话总结

1. **每个神经元都有自己专属的一条 face axis 和一条 object axis**，它们是由该神经元自己的 firing rate 对 60-D 图像特征做线性回归得到的权重向量 **c**，**不是**"把图像特征平均"得到的。
2. Face 轴和 Object 轴**共享同一个 60-D 坐标系**（general-object space），但用**不同的图片子集**（分别是 face 或 object）的 r 和 F 回归得到，因此可以在同一个空间中计算它们之间的 cosine。
3. **Fig. 3e** 的每一行是一个 cell，每一列是一个 latency T，颜色 = 该 cell 的 **time-varying axis(T)** 与该 cell 的 **overall object axis (50–220 ms)** 的 cosine similarity；top panel 用 time-varying **face** axis，bottom panel 用 time-varying **object** axis。

---

**User**

好的，fig2中的face和object axis是如何得到的

---

**Cursor**

Fig. 2 其实是**最简单**的 axis 分析——它只用 **50–220 ms 整个窗口平均**的一条 axis，没有时间维度。整体思路和 Fig. 3 的 "overall axis" 完全一致，只是 Fig. 3 在其基础上又加了 time-varying axis。下面按面板把 Fig. 2 全部讲清楚。

---

## 一、Fig. 2 所用的 axis 拟合流程

对于 **每一个入选 cell（n = 151, ML of monkey A）**，做**两次**独立的线性回归：

### (1) 准备（所有 cell 共用）
- 60-D 特征空间 = **general-object space**（AlexNet fc6 → PCA(100 faces + 1,292 objects) → 取前 60 PC → 每维 z-score）
- 每张图都有一个 60-D 特征向量 **f**_j
- 图片分 train/test：1,292 个物体中 1,192 train + 100 test；1,425 张脸中 1,325 train + 100 test（对应原文"training set 重复 1 次、test set 重复 3 次"，第 222 行）

### (2) Object axis（对 cell i）
- r_i^{obj} = cell i 对 **1,192 个训练物体**在 **50–220 ms 平均** firing rate（所有 trial 平均）
- F_obj_train ∈ ℝ^{1192 × 60}
- 解 OLS：
  \[
  \mathbf{c}_i^{\text{obj}} = (\mathbf{r}_i^{\text{obj}} - \bar r) \, F_{\text{obj,train}} \, (F_{\text{obj,train}}^T F_{\text{obj,train}})^{-1}
  \]
- 用 100 个 test 物体评估 R²_{obj,test}

### (3) Face axis（对同一个 cell i）
- r_i^{face} = cell i 对 **1,325 张训练脸**在 50–220 ms 平均 firing rate
- F_face_train ∈ ℝ^{1325 × 60}（**同一套 60 个 PC**，只是图行换成脸）
- 同样 OLS 得到 **c**_i^{face}
- 用 100 张 test 脸评估 R²_{face,test}

### (4) 入选条件
R²_{obj,test} > 0 **且** R²_{face,test} > 0 → 保留该 cell。ML of monkey A 有 151/563 满足，这就是 Fig. 2 的 n = 151。

每个 cell 由此得到 **两个 60-D 向量 c_i^{obj}, c_i^{face}**，全部画在 Fig. 2 的四个面板里。

---

## 二、Fig. 2 每个面板画的是什么

### Fig. 2a（两张极坐标/箭头图）

**目的**：比较所有 cell 的 object 轴朝向分布 vs face 轴朝向分布。

- 取每条 60-D 轴 **c**_i 的前两个分量 (c_{i,1}, c_{i,2})——即在 PC1–PC2 平面上的投影
- 每个 cell 画一根从原点出发的箭头（或一个方向角）
- **左图（绿）**：151 条 object 轴 → 集中指向右上象限（脸所处的区域），角度 s.d. = 37.4°
- **右图（紫）**：151 条 face 轴 → 朝向分散，角度 s.d. = 81.9°

结论：object 轴非常一致（face cell 对物体的编码方向同质），face 轴发散（每个 cell 对脸的编码方向各异）。

### Fig. 2b（两张散点图）

**目的**：在单个 PC 维度上直接比 object 轴权重 vs face 轴权重是否相关。

- 左图：x = c_i^{obj}[PC1], y = c_i^{face}[PC1]，每个 cell 一个点；r = 0.16
- 右图：x = c_i^{obj}[PC2], y = c_i^{face}[PC2]，每个 cell 一个点；r = −0.07

两个维度上都看不到相关 → face 和 object 轴不是同一方向。

### Fig. 2c（cross-prediction R² 分布）

**目的**：从预测的角度证实两条轴不同。

- 左图（用来解释 **物体响应**）：
  - **灰色分布**：用 **face 轴** c_i^{face} 去预测 cell i 对 test 物体的响应，R² 分布（几乎全 ≤ 0）
  - **绿色分布**：把 1,292 物体随机分两半，用一半拟物体轴，用另一半算 R²（同类预测的"天花板"）
- 右图（用来解释 **脸响应**）：对称构造，灰色 = 用 object 轴预测脸响应，紫色 = face 轴 split-half 预测脸响应

跨类别预测（灰）几乎无效；同类别预测（彩）明显为正 → 两条轴真的是独立的编码方向。

### Fig. 2d（一个 example cell）

- 顶左：该 cell 对 100 test 物体的真实响应 vs **c_i^{obj} · f_j** 的预测，R² = 0.62
- 顶右：该 cell 对 100 test 脸的真实响应 vs **c_i^{face} · f_j** 的预测，R² = 0.54
- 底：把该 cell 所有图片画在 PC1–PC2 平面，物体用绿色+箭头（c_i^{obj}），脸用紫色+箭头（c_i^{face}）；两根箭头**指向近乎相反**的方向

直观展示"同一个细胞，脸轴和物体轴确实可以完全反向"。

---

## 三、Fig. 2 vs Fig. 3 的差别一眼看清

| 维度 | Fig. 2 | Fig. 3 |
|---|---|---|
| 特征空间 | 60-D general-object space | 同上（一模一样） |
| 时间窗口 | **一个**：50–220 ms 平均 | **一串**：20 ms 滑动窗口 + 50–220 ms overall 做参考 |
| 每个 cell 得到的 axis 数 | 2 条（c^{obj}, c^{face}） | 2 条 overall + 2×N_T 条 time-varying |
| 主要展示 | 轴方向的分布 / 两类轴是否相关 | 轴方向**随时间的变化**，特别是 face 轴的翻转 |

换句话说，**Fig. 3 的 "overall axis" 就是 Fig. 2 用的 axis**；Fig. 3 只是在它之上再加了 time-varying axis，然后把两者做 cosine similarity 画成热图。

---

## 四、伪代码对照

```python
def fit_axis(F, r):
    r_c = r - r.mean()
    return np.linalg.solve(F.T @ F, F.T @ r_c)   # (60,)

for i in range(n_cells):
    r_obj  = fr[i, obj_train,  50:220].mean(axis=time_axis)   # (1192,)
    r_face = fr[i, face_train, 50:220].mean(axis=time_axis)   # (1325,)
    c_obj[i]  = fit_axis(F_obj[obj_train],   r_obj)
    c_face[i] = fit_axis(F_face[face_train], r_face)

    r_obj_test  = fr[i, obj_test,  50:220].mean(...)
    r_face_test = fr[i, face_test, 50:220].mean(...)
    R2_obj[i]  = r2_score(r_obj_test,  F_obj[obj_test]   @ c_obj[i])
    R2_face[i] = r2_score(r_face_test, F_face[face_test] @ c_face[i])

keep = (R2_obj > 0) & (R2_face > 0)
c_obj, c_face = c_obj[keep], c_face[keep]   # n = 151

# ---- Fig. 2a 分布图 ----
plot_arrows_on_PC1PC2(c_obj[:, :2], color='green')
plot_arrows_on_PC1PC2(c_face[:, :2], color='purple')

# ---- Fig. 2b 散点 ----
scatter(c_obj[:, 0], c_face[:, 0])   # PC1
scatter(c_obj[:, 1], c_face[:, 1])   # PC2

# ---- Fig. 2c cross prediction ----
# 用 c_face 预测 object test 响应的 R²，vs 用 c_obj(split-half) 自预测
```

---

## 五、一句话概括

> Fig. 2 的 face/object axis = **每个 cell 用自己 50–220 ms 的 mean firing rate**，在**共享的 60-D general-object space** 里，对 face 图子集和 object 图子集**各做一次线性回归**得到的两条 60-D 权重向量；Fig. 2 的所有面板都是在这两条 per-cell 轴上做的方向分布、权重散点和 cross-prediction 分析。

如果您想把它复现到 `A21_PCA_Cross_Decode.py` / `A3_Cross_Decode.py`，可以告诉我当前脚本里已有哪些变量（`fr`, `F`, stim 索引等），我可以直接对照您的数据结构写出严谨的实现。

