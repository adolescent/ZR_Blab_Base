'''
Decode shuffle level (continuous 0-4) with linear SVR, per image.
Fixed-time feature: avr_rsp.npy (170 ms window), per area, all / ani / inani scopes.
LORO: leave one repeat out across 5 shuffles (25 samples per image).
'''

#%%
import numpy as np
import pandas as pd
import OS_Tools as ot
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

savepath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\SVM_Decoding'
datapath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Raw_Metamer_1k'
brain_areas = ['ML', 'MSB', 'AL', 'ASB']

WINDOW_S = 0.17
N_REPEAT = 5
N_SHUF = 5
N_IMG = 40
N_COND = N_SHUF * N_IMG
N_COL = N_REPEAT * N_COND
CHANCE_MAE = np.mean([abs(i - j) for i in range(N_SHUF) for j in range(N_SHUF)])

SCOPES = [
    ('all', slice(0, 40)),
    ('ani', slice(0, 20)),
    ('inani', slice(20, 40)),
]

#%%


def load_area(area):
    area_dir = ot.Join(datapath, area)
    rsp_hz = np.load(ot.Join(area_dir, 'avr_rsp.npy')) / WINDOW_S
    if rsp_hz.shape[1] != N_COL:
        raise ValueError(f'{area}: expected {N_COL} columns, got {rsp_hz.shape[1]}')
    return rsp_hz


def build_by_shuffle(rsp_hz):
    """dict[shuffle] -> (n_cell, 40, 5); axis1=img_id, axis2=repeat."""
    n_cell = rsp_hz.shape[0]
    r2 = rsp_hz.reshape(n_cell, N_REPEAT, N_COND)
    out = {}
    for s in range(N_SHUF):
        out[s] = r2[:, :, s * N_IMG : (s + 1) * N_IMG].transpose(0, 2, 1)
    return out


def get_X_y_shuffle(by_shuf, img_idx):
    """One image: X (25, n_cell), y (25,) continuous shuffle level, groups repeat index."""
    X = np.vstack([by_shuf[s][:, img_idx, :].T for s in range(N_SHUF)])
    y = np.repeat(np.arange(N_SHUF), N_REPEAT).astype(float)
    groups = np.tile(np.arange(N_REPEAT), N_SHUF)
    return X, y, groups


def predict_shuffle_loro(X, y, groups):
    """LORO over repeats; return pooled y_true, y_pred on all test folds."""
    reg = make_pipeline(StandardScaler(), SVR(kernel='linear'))
    y_true_all, y_pred_all = [], []
    for repeat in range(N_REPEAT):
        te = groups == repeat
        tr = ~te
        reg.fit(X[tr], y[tr])
        y_pred_all.append(reg.predict(X[te]))
        y_true_all.append(y[te])
    return np.concatenate(y_true_all), np.concatenate(y_pred_all)


def decode_shuffle_loro(X, y, groups):
    """LORO over repeats; return list of (repeat, abs_error) on test fold."""
    reg = make_pipeline(StandardScaler(), SVR(kernel='linear'))
    out = []
    for repeat in range(N_REPEAT):
        te = groups == repeat
        tr = ~te
        reg.fit(X[tr], y[tr])
        y_pred = reg.predict(X[te])
        mae = float(np.abs(y_pred - y[te]).mean())
        out.append((repeat, mae))
    return out


#%%
rows = []
for area in tqdm(brain_areas, desc='area'):
    rsp_hz = load_area(area)
    by_shuf = build_by_shuffle(rsp_hz)
    n_cell = rsp_hz.shape[0]

    for scope_name, img_slice in SCOPES:
        for img_idx in range(img_slice.start, img_slice.stop):
            X, y, groups = get_X_y_shuffle(by_shuf, img_idx)
            for repeat, mae in decode_shuffle_loro(X, y, groups):
                rows.append({
                    'area': area,
                    'scope': scope_name,
                    'img_id': img_idx,
                    'repeat': repeat,
                    'abs_error': mae,
                    'n_shuffle': N_SHUF,
                    'chance_mae': CHANCE_MAE,
                    'n_cell': n_cell,
                })

df = pd.DataFrame(rows)
ot.Mkdir(savepath, mute=True)
out_csv = ot.Join(savepath, 'shuffle_level_decode_loo.csv')
df.to_csv(out_csv, index=False)

print(f'Saved {len(df)} rows -> {out_csv}')
print(df.groupby(['area', 'scope']).size())
n_expected = len(brain_areas) * sum(s.stop - s.start for _, s in SCOPES) * N_REPEAT
assert len(df) == n_expected
assert df.groupby(['area', 'scope', 'img_id']).size().eq(N_REPEAT).all()

#%%
import matplotlib.pyplot as plt
import seaborn as sns

# df = pd.read_csv(out_csv)  # uncomment if running this cell alone

fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
sns.boxplot(
    data=df, x='area', y='abs_error', hue='scope', order=brain_areas,
    ax=ax, showfliers=False,
)
ax.axhline(CHANCE_MAE, linestyle='--', color='gray', alpha=0.5)
ax.set_ylim(0, 1.5)
ax.set_xlabel('')
ax.set_ylabel('Mean abs error')
ax.set_title('Shuffle level decode — SVR (LORO)')
plt.tight_layout()
plt.show()

#%%
cm_scope = 'all'     # all, ani, inani


def confusion_pct(area):
    rsp_hz = load_area(area)
    by_shuf = build_by_shuffle(rsp_hz)
    y_true_all, y_pred_all = [], []
    for scope_name, img_slice in SCOPES:
        if scope_name != cm_scope:
            continue
        for img_idx in range(img_slice.start, img_slice.stop):
            X, y, groups = get_X_y_shuffle(by_shuf, img_idx)
            y_true, y_pred = predict_shuffle_loro(X, y, groups)
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)
    y_true = np.concatenate(y_true_all)
    y_pred_disc = np.rint(np.concatenate(y_pred_all)).clip(0, N_SHUF - 1).astype(int)
    cm_count = np.zeros((N_SHUF, N_SHUF), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred_disc):
        cm_count[t, p] += 1
    return cm_count / cm_count.sum(axis=1, keepdims=True) * 100


levels = list(range(N_SHUF))
fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=150)
for ax, area in zip(axes.ravel(), brain_areas):
    sns.heatmap(
        confusion_pct(area), annot=True, fmt='.1f', cmap='Blues', vmin=0, vmax=100,square=True,
        xticklabels=levels, yticklabels=levels, ax=ax, cbar=False,
    )
    ax.set_title(area)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
fig.suptitle(f'Confusion matrix — {cm_scope}')
plt.tight_layout()
plt.show()

#%%



