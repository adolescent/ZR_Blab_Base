'''
Decode image ID (20/40-class) with linear SVM; test generalization across shuffle levels.
Fixed-time feature: avr_rsp.npy (170 ms window), per area, all / ani / inani scopes.
'''

#%%
import numpy as np
import pandas as pd
import OS_Tools as ot
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

savepath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\SVM_Decoding'
datapath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Raw_Metamer_1k'
brain_areas = ['ML', 'MSB', 'AL', 'ASB']

WINDOW_S = 0.17
N_REPEAT = 5
N_SHUF = 5
N_IMG = 40
N_COND = N_SHUF * N_IMG
N_COL = N_REPEAT * N_COND

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


def get_X_y(resp_sh, img_slice):
    """resp_sh: (n_cell, 40, 5) -> X (n_class*5, n_cell), y (n_class*5,)."""
    r = resp_sh[:, img_slice, :]
    n_class = r.shape[1]
    X = r.transpose(1, 2, 0).reshape(-1, r.shape[0])
    y = np.repeat(np.arange(n_class), N_REPEAT)
    return X, y


def get_X_y_repeat(resp_sh, img_slice, repeat):
    """One repeat block: X (n_class, n_cell), y (n_class,)."""
    r = resp_sh[:, img_slice, repeat]
    return r.T, np.arange(r.shape[1])


def decode_one(train_sh, test_sh, X_train, y_train, by_shuf, img_slice):
    """Return list of (repeat, correct_rate) — 5 rows per train/test shuffle pair."""
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    out = []
    if train_sh == test_sh:
        groups = np.arange(len(y_train)) % N_REPEAT
        for repeat in range(N_REPEAT):
            te = groups == repeat
            tr = ~te
            clf.fit(X_train[tr], y_train[tr])
            cr = float((clf.predict(X_train[te]) == y_train[te]).mean())
            out.append((repeat, cr))
    else:
        for repeat in range(N_REPEAT):
            X_tr, y_tr = get_X_y_repeat(by_shuf[train_sh], img_slice, repeat)
            X_te, y_te = get_X_y_repeat(by_shuf[test_sh], img_slice, repeat)
            clf.fit(X_tr, y_tr)
            cr = float((clf.predict(X_te) == y_te).mean())
            out.append((repeat, cr))
    return out


#%%
rows = []
for area in tqdm(brain_areas, desc='area'):
    rsp_hz = load_area(area)
    n_cell = rsp_hz.shape[0]
    by_shuf = build_by_shuffle(rsp_hz)

    for scope_name, img_slice in SCOPES:
        n_class = img_slice.stop - img_slice.start
        for train_sh in range(N_SHUF):
            X_train, y_train = get_X_y(by_shuf[train_sh], img_slice)
            for test_sh in range(N_SHUF):
                for repeat, cr in decode_one(
                    train_sh, test_sh, X_train, y_train, by_shuf, img_slice
                ):
                    rows.append({
                        'area': area,
                        'scope': scope_name,
                        'n_class': n_class,
                        'train_shuffle': train_sh,
                        'test_shuffle': test_sh,
                        'repeat': repeat,
                        'same_shuffle': train_sh == test_sh,
                        'correct_rate': cr,
                        'n_cell': n_cell,
                    })

df = pd.DataFrame(rows)
ot.Mkdir(savepath, mute=True)
out_csv = ot.Join(savepath, 'figid_decode_cross_shuffle.csv')
df.to_csv(out_csv, index=False)

print(f'Saved {len(df)} rows -> {out_csv}')
print(df.groupby(['area', 'scope']).size())
n_combo = N_SHUF * N_SHUF * N_REPEAT
assert len(df) == len(brain_areas) * len(SCOPES) * n_combo
assert df.groupby(['area', 'scope']).size().eq(n_combo).all()
assert df.groupby(['area', 'scope', 'train_shuffle', 'test_shuffle']).size().eq(N_REPEAT).all()
assert df['same_shuffle'].sum() == len(brain_areas) * len(SCOPES) * N_SHUF * N_REPEAT

#%%
import matplotlib.pyplot as plt
import seaborn as sns

# df = pd.read_csv(out_csv)  # uncomment if running this cell alone
heatmap_scope = 'inani'    # all, ani, inani

fig, axes = plt.subplots(2, 2, figsize=(9, 8), dpi=150)
for ax, area in zip(axes.ravel(), brain_areas):
    sub = df.loc[(df['area'] == area) & (df['scope'] == heatmap_scope)]
    mat = (
        sub.groupby(['train_shuffle', 'test_shuffle'], as_index=False)['correct_rate']
        .mean()
        .pivot(index='train_shuffle', columns='test_shuffle', values='correct_rate')
        .reindex(index=range(N_SHUF), columns=range(N_SHUF))
    )
    sns.heatmap(
        mat, ax=ax, annot=True, fmt='.2f', cmap='viridis',
        vmin=0, vmax=1, square=True, cbar=False,
    )
    ax.set_title(area)
    ax.set_xlabel('Test shuffle')
    ax.set_ylabel('Train shuffle')

n_class = df.loc[df['scope'] == heatmap_scope, 'n_class'].iloc[0]
fig.suptitle(f'FigID decode — {heatmap_scope} (n_class={n_class})')
plt.tight_layout()
plt.show()

#%%
# Calculate shuffle-distance sensitivity and directionality metrics.
# This cell only computes and saves tables; plotting is separated below.
metric_rows = []
curve_rows = []

for (area, scope, repeat), sub in df.groupby(['area', 'scope', 'repeat']):
    n_class = int(sub['n_class'].iloc[0])
    chance = 1 / n_class
    sub = sub.copy()
    sub['correct_rate_chance_corr'] = (
        sub['correct_rate'] - chance
    ) / (1 - chance)
    sub['shuffle_distance'] = (
        sub['test_shuffle'] - sub['train_shuffle']
    ).abs()
    sub['signed_distance'] = sub['test_shuffle'] - sub['train_shuffle']

    dist_mean = (
        sub.groupby('shuffle_distance', as_index=False)['correct_rate_chance_corr']
        .mean()
        .rename(columns={'correct_rate_chance_corr': 'mean_acc_corr'})
    )
    g0 = float(dist_mean.loc[dist_mean['shuffle_distance'] == 0, 'mean_acc_corr'].iloc[0])
    dist_mean['relative_acc'] = dist_mean['mean_acc_corr'] / g0 if g0 > 0 else np.nan

    for _, row in dist_mean.iterrows():
        curve_rows.append({
            'area': area,
            'scope': scope,
            'repeat': repeat,
            'n_class': n_class,
            'shuffle_distance': int(row['shuffle_distance']),
            'mean_acc_corr': float(row['mean_acc_corr']),
            'relative_acc': float(row['relative_acc']),
        })

    valid_rel = np.isfinite(dist_mean['relative_acc'])
    rel_slope = (
        np.polyfit(
            dist_mean.loc[valid_rel, 'shuffle_distance'],
            dist_mean.loc[valid_rel, 'relative_acc'],
            1,
        )[0]
        if valid_rel.sum() >= 2
        else np.nan
    )
    abs_slope = np.polyfit(
        dist_mean['shuffle_distance'],
        dist_mean['mean_acc_corr'],
        1,
    )[0]

    mat = (
        sub.pivot(index='train_shuffle', columns='test_shuffle', values='correct_rate_chance_corr')
        .reindex(index=range(N_SHUF), columns=range(N_SHUF))
    )
    direction_diffs = []
    for i in range(N_SHUF):
        for j in range(i + 1, N_SHUF):
            direction_diffs.append(mat.loc[i, j] - mat.loc[j, i])

    metric_rows.append({
        'area': area,
        'scope': scope,
        'repeat': repeat,
        'n_class': n_class,
        'diag_acc_corr': g0,
        'offdiag_acc_corr': float(sub.loc[sub['shuffle_distance'] > 0, 'correct_rate_chance_corr'].mean()),
        'shuffle_sensitivity_rel': float(-rel_slope),
        'shuffle_sensitivity_abs': float(-abs_slope),
        'direction_bias_up_minus_down': float(np.mean(direction_diffs)),
        'direction_asymmetry_abs': float(np.mean(np.abs(direction_diffs))),
    })

distance_curve_df = pd.DataFrame(curve_rows)
shuffle_metric_df = pd.DataFrame(metric_rows)

distance_curve_csv = ot.Join(savepath, 'figid_shuffle_distance_curve.csv')
shuffle_metric_csv = ot.Join(savepath, 'figid_shuffle_sensitivity_metrics.csv')
distance_curve_df.to_csv(distance_curve_csv, index=False)
shuffle_metric_df.to_csv(shuffle_metric_csv, index=False)

print(f'Saved distance curve -> {distance_curve_csv}')
print(f'Saved sensitivity metrics -> {shuffle_metric_csv}')
print(shuffle_metric_df.groupby(['area', 'scope']).size())

#%%
# Plot distance sensitivity and directionality. Adjust scope/figure settings here.
metric_scope = 'ani'    # all, ani, inani

curve_plot = distance_curve_df.loc[distance_curve_df['scope'] == metric_scope]
metric_plot = shuffle_metric_df.loc[shuffle_metric_df['scope'] == metric_scope]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

sns.lineplot(
    data=curve_plot,
    x='shuffle_distance',
    y='relative_acc',
    hue='area',
    marker='o',
    ax=axes[0],
)
axes[0].axhline(1, linestyle='--', color='gray', alpha=0.5)
axes[0].set_xticks(range(N_SHUF))
axes[0].set_xlabel('|Train shuffle - test shuffle|')
axes[0].set_ylabel('Relative chance-corrected accuracy')
axes[0].set_title('Shuffle distance decay')

sns.barplot(
    data=metric_plot,
    x='area',
    y='direction_bias_up_minus_down',
    ax=axes[1],
)
axes[1].axhline(0, linestyle='--', color='gray', alpha=0.5)
axes[1].set_xlabel('Area')
axes[1].set_ylabel('Up - down accuracy')
axes[1].set_title('Direction bias')

fig.suptitle(f'cross-shuffle generalization — {metric_scope}')
plt.tight_layout()
plt.show()

#%%



