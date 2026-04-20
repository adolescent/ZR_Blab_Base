'''
Use cell category in A1, and do time-windowed corr between different shufle level, making it 

'''


#%%
from Py_Structure.Info_Files.Stim_ID_List import Stim_ID
import OS_Tools as ot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as JL
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

Stim_Caller = Stim_ID(stim_type='Metamer_Raw')
metamer_ids = Stim_Caller.Stim_Conditions
datafolder = r'E:\#Preprocessed_Data\Selected_Cells'
wp= r'E:\#Preprocessed_Data\260402_TC_Analysis'


#%% data loading and pre-processing.
brain_areas = ['MSB','ASB','AL','ML']
used_ares = brain_areas[3]

psths = np.load(ot.Join(datafolder,f'{used_ares}_Cells_Metamer_Only.npz'))['psth']

# 0–300 ms → bins [100:400]; joint PMF over (img, time): denom = sum over 1000 imgs and 300 time pts
t_ms0, t_ms1 = 0, 300
t_idx0, t_idx1 = t_ms0 + 100, t_ms1 + 100
assert t_idx1 - t_idx0 == (t_ms1 - t_ms0)
resp_t = psths[:, :, t_idx0:t_idx1]
n_img, n_t = resp_t.shape[1], resp_t.shape[2]
den = resp_t.sum(axis=(1, 2), keepdims=True)
psth_pdf = np.divide(resp_t, den, out=np.zeros_like(resp_t, dtype=float), where=den > 0) * 1000


#%%
cell_classes =np.load(ot.Join(wp,f'{used_ares}_pca_kmeans3_ztime_0to250ms.npz'))['labels']
cells_c0 = psth_pdf[cell_classes==0,:,:]
cells_c1 = psth_pdf[cell_classes==1,:,:]
cells_c2 = psth_pdf[cell_classes==2,:,:]
c0_avr = cells_c0.mean(0)
c1_avr = cells_c1.mean(0)
c2_avr = cells_c2.mean(0)

#%% ##################### Analysis 1, calculate corr between shuffle levels.
win_len = 10
win_step = 5
img_range = np.arange(1,41)

# Trial row 0..999 aligns with metamer_ids row order and psths axis 1
_meta = metamer_ids.assign(trial_row=np.arange(len(metamer_ids)))
trial_indices = {}
for (img, shuf), grp in _meta.groupby(['Img_Index', 'Shuffle_Level'], sort=False):
    trial_indices[(int(img), int(shuf))] = grp['trial_row'].to_numpy(dtype=int)
for k, v in trial_indices.items():
    assert len(v) == 5, (k, len(v))


def pearson_safe(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2 or b.size < 2:
        return np.nan
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else np.nan


def iter_rep_pairs(same_shuffle_level):
    if same_shuffle_level:
        for ia in range(5):
            for ib in range(ia + 1, 5):
                yield ia, ib
    else:
        for ia in range(5):
            for ib in range(5):
                yield ia, ib


def corr_shuffle_time_windows(resp, trial_indices, img, shuf_a, shuf_b, wa, wb, win_len):
    """resp: (n_cells, 1000, n_t) psth_pdf slice; window indices wa, wb relative to 0–300 ms slice."""
    idx_a = trial_indices[(int(img), int(shuf_a))]
    idx_b = trial_indices[(int(img), int(shuf_b))]
    # Exclude self-pairs only when data are exactly same source: same shuffle and same window
    if int(shuf_a) == int(shuf_b):
        same = abs(int(wa) - int(wb))<=5 # within 2 steps of each other.
    else:
        same = False
    # same = False
    rs = []
    for ia, ib in iter_rep_pairs(same):
        v1 = resp[:, idx_a[ia], wa : wa + win_len].sum(axis=-1)
        v2 = resp[:, idx_b[ib], wb : wb + win_len].sum(axis=-1)
        r = pearson_safe(v1, v2)
        if np.isfinite(r):
            rs.append(r)
    n_expect = 10 if same else 25
    return (float(np.mean(rs)) if rs else np.nan), n_expect, len(rs)


def build_shuffle_timecorr_table(
    resp_pdf,
    trial_indices,
    t_ms0,
    win_len,
    win_step,
    img_range,
    brain_area,
    cell_group_name,
):
    """resp_pdf: (n_cells, 1000, n_t) e.g. cells_c0 from psth_pdf[cell_classes == 0]."""
    n_cells = int(resp_pdf.shape[0])
    n_t = resp_pdf.shape[2]
    win_starts = list(range(0, n_t - win_len + 1, win_step))
    rows = []
    for img in tqdm(img_range, total=len(img_range)):
        for shuf_a in range(5):
            for shuf_b in range(5):
                for wa in win_starts:
                    for wb in win_starts:
                        c_mean, n_pairs, n_valid = corr_shuffle_time_windows(
                            resp_pdf, trial_indices, int(img), shuf_a, shuf_b, wa, wb, win_len
                        )
                        rows.append(
                            {
                                'brain_area': brain_area,
                                'cell_group': cell_group_name,
                                'n_cells': n_cells,
                                'Img_Index': int(img),
                                'shuffle_level_A': shuf_a,
                                'shuffle_level_B': shuf_b,
                                'win_start_A': wa,
                                'win_start_B': wb,
                                'win_end_A': wa + win_len,
                                'win_end_B': wb + win_len,
                                't_ms_A_start': t_ms0 + wa,
                                't_ms_A_end': t_ms0 + wa + win_len,
                                't_ms_B_start': t_ms0 + wb,
                                't_ms_B_end': t_ms0 + wb + win_len,
                                'win_len': win_len,
                                'corr_mean': c_mean,
                                'n_pairs': n_pairs,
                                'n_valid_corr': n_valid,
                            }
                        )
    return pd.DataFrame(rows)


group_defs = [
    ('c0', cells_c0),
    ('c1', cells_c1),
    ('c2', cells_c2),
]
shuffle_timecorr_frames = []
for name, resp_pdf in group_defs:
    if resp_pdf.shape[0] == 0:
        continue
    shuffle_timecorr_frames.append(
        build_shuffle_timecorr_table(
            resp_pdf,
            trial_indices,
            t_ms0,
            win_len,
            win_step,
            img_range,
            used_ares,
            name,
        )
    )
shuffle_timecorr_df = None
if shuffle_timecorr_frames:
    shuffle_timecorr_df = pd.concat(shuffle_timecorr_frames, ignore_index=True)
    out_parquet = ot.Join(wp, f'{used_ares}_shuffle_timecorr_psth_pdf_win{win_len}_step{win_step}.parquet')
    shuffle_timecorr_df.to_parquet(out_parquet, index=False)


#%% Viz: correlation heatmap over time-window pairs (fixed image + shuffle A vs B)
def plot_shuffle_timecorr_heatmap(
    df,
    img_index,
    shuffle_a,
    shuffle_b,
    cell_group='c0',
    t_ms_offset=0,
    tick_ms_step=60,
    plot_t_start_ms=None,
    plot_t_end_ms=None,
    figsize=(8, 8),
    save_path=None,vmax=1.0
):
    """Heatmap: x = window B start (ms), y = window A start (ms), value = corr_mean.
    Rows are flipped so smaller A is toward the bottom (Cartesian-style with time).
    Axis tick *labels* only at multiples of tick_ms_step (relative window start, default 60 ms).
    Draws a light line along win_start_A == win_start_B for comparing above/below diagonal.
    Use plot_t_start_ms/plot_t_end_ms to crop displayed time-window starts (absolute ms)."""
    sub = df.loc[
        (df['Img_Index'] == int(img_index))
        & (df['shuffle_level_A'] == int(shuffle_a))
        & (df['shuffle_level_B'] == int(shuffle_b))
        & (df['cell_group'] == str(cell_group))
    ]
    toff = int(t_ms_offset)
    if plot_t_start_ms is not None:
        rel_start = int(plot_t_start_ms) - toff
        sub = sub.loc[(sub['win_start_A'] >= rel_start) & (sub['win_start_B'] >= rel_start)]
    if plot_t_end_ms is not None:
        rel_end = int(plot_t_end_ms) - toff
        sub = sub.loc[(sub['win_start_A'] <= rel_end) & (sub['win_start_B'] <= rel_end)]
    if sub.empty:
        raise ValueError(
            'No rows for this Img_Index / shuffle A,B / cell_group / plot time range. '
            'Check img_range and plot_t_start_ms / plot_t_end_ms.'
        )
    pivot = sub.pivot_table(
        index='win_start_A',
        columns='win_start_B',
        values='corr_mean',
        aggfunc='first',
    )
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    n_cells = int(sub['n_cells'].iloc[0])
    wl = int(sub['win_len'].iloc[0])
    # Flip rows so larger win_start_A is at top, smaller at bottom (early A = bottom, like a time axis)
    pivot_plot = pivot.iloc[::-1, :]
    rel_a = pivot_plot.index.to_numpy(dtype=int)
    rel_b = pivot.columns.to_numpy(dtype=int)
    step = int(tick_ms_step)

    def _sparse_ms_labels(rel_starts):
        return [
            f'{toff + int(r)}' if int(r) % step == 0 else ''
            for r in rel_starts
        ]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot_plot.to_numpy(dtype=float),
        ax=ax,
        cmap='coolwarm',
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        square=True,
        xticklabels=_sparse_ms_labels(rel_b),
        yticklabels=_sparse_ms_labels(rel_a),
        cbar_kws={
            'label': 'mean Pearson r',
            'shrink': 0.55,
            'aspect': 28,
        },
    )
    # Same-time diagonal (wa == wb): cell centers (j+0.5, i+0.5) in heatmap data coords
    row_labels = pivot_plot.index.to_numpy(dtype=int)
    col_labels = pivot_plot.columns.to_numpy(dtype=int)
    row_pos = {int(v): i for i, v in enumerate(row_labels)}
    col_pos = {int(v): j for j, v in enumerate(col_labels)}
    common_starts = sorted(set(row_pos.keys()) & set(col_pos.keys()))
    if common_starts:
        xs = [col_pos[v] + 0.5 for v in common_starts]
        ys = [row_pos[v] + 0.5 for v in common_starts]
        ax.plot(
            xs,
            ys,
            color=(0.92, 0.92, 0.92),
            linewidth=1.4,
            linestyle='-',
            zorder=5,
            clip_on=True,
        )
    # Square axes for symmetric comparison of upper vs lower triangle
    ax.set_aspect('equal', adjustable='box')
    # Columns = win_start_B (left→right); rows = win_start_A (bottom→top after flip)
    ax.set_xlabel(f'Time window B start (ms)\n(win_len={wl}; shuffle_level_B={int(shuffle_b)})')
    ax.set_ylabel(f'Time window A start (ms)\n(win_len={wl}; shuffle_level_A={int(shuffle_a)})')
    ax.set_title(
        f'{sub["brain_area"].iloc[0]} | {cell_group} (n={n_cells} cells)\n'
        f'Img_Index={int(img_index)} | shuffle A={int(shuffle_a)} vs B={int(shuffle_b)}'
    )
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


plot_img_index = 9
plot_shuffle_a = 0
plot_shuffle_b = 4
plot_cell_group = 'c2'
# If you run only this cell: load parquet (match win_len/step/filename used when saving):
# shuffle_timecorr_df = pd.read_parquet(ot.Join(wp, f'{used_ares}_shuffle_timecorr_psth_pdf_win{win_len}_step{win_step}.parquet'))

if shuffle_timecorr_df is not None:
    _heatmap_path = ot.Join(
        wp,
        f'{used_ares}_heatmap_img{plot_img_index}_Sh{plot_shuffle_a}v{plot_shuffle_b}_{plot_cell_group}.png',
    )
    plot_shuffle_timecorr_heatmap(
        shuffle_timecorr_df,
        plot_img_index,
        plot_shuffle_a,
        plot_shuffle_b,
        cell_group=plot_cell_group,
        t_ms_offset=t_ms0,
        tick_ms_step=20,
        # plot_t_start_ms=40,
        plot_t_end_ms=220,
        save_path=None,vmax=0.2
    )
    plt.show()

#%%

all_img_corr = pd.read_parquet(
    ot.Join(wp, f'{used_ares}_shuffle_timecorr_psth_pdf_win10_step5.parquet')
)

plot_shuffle_a = 0
plot_shuffle_b = 4
plot_cell_group = 'c1'
vmax=0.1
avg_img_indices = np.arange(1, 21)

def plot_shuffle_timecorr_heatmap_mean_imgs(
    df,
    img_indices,
    shuffle_a,
    shuffle_b,
    cell_group='c0',
    t_ms_offset=0,
    tick_ms_step=60,
    plot_t_start_ms=None,
    plot_t_end_ms=None,
    figsize=(8, 8),
    save_path=None,
    vmax=1.0,
):
    """Average corr_mean over Img_Index, then same heatmap as plot_shuffle_timecorr_heatmap."""
    img_indices = np.unique(np.asarray(img_indices, dtype=int).ravel())
    sub = df.loc[
        df['Img_Index'].isin(img_indices)
        & (df['shuffle_level_A'] == int(shuffle_a))
        & (df['shuffle_level_B'] == int(shuffle_b))
        & (df['cell_group'] == str(cell_group))
    ]
    toff = int(t_ms_offset)
    if plot_t_start_ms is not None:
        rel_start = int(plot_t_start_ms) - toff
        sub = sub.loc[(sub['win_start_A'] >= rel_start) & (sub['win_start_B'] >= rel_start)]
    if plot_t_end_ms is not None:
        rel_end = int(plot_t_end_ms) - toff
        sub = sub.loc[(sub['win_start_A'] <= rel_end) & (sub['win_start_B'] <= rel_end)]
    if sub.empty:
        raise ValueError(
            'No rows for img_indices / shuffle A,B / cell_group / plot time range.'
        )
    n_imgs_used = int(sub['Img_Index'].nunique())
    agg = sub.groupby(['win_start_A', 'win_start_B'], as_index=False).agg(
        corr_mean=('corr_mean', 'mean'),
        n_cells=('n_cells', 'first'),
        win_len=('win_len', 'first'),
        brain_area=('brain_area', 'first'),
    )
    pivot = agg.pivot_table(
        index='win_start_A',
        columns='win_start_B',
        values='corr_mean',
        aggfunc='first',
    )
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    n_cells = int(agg['n_cells'].iloc[0])
    wl = int(agg['win_len'].iloc[0])
    brain_area = str(agg['brain_area'].iloc[0])
    pivot_plot = pivot.iloc[::-1, :]
    rel_a = pivot_plot.index.to_numpy(dtype=int)
    rel_b = pivot.columns.to_numpy(dtype=int)
    step = int(tick_ms_step)

    def _sparse_ms_labels(rel_starts):
        return [
            f'{toff + int(r)}' if int(r) % step == 0 else ''
            for r in rel_starts
        ]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot_plot.to_numpy(dtype=float),
        ax=ax,
        cmap='coolwarm',
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        square=True,
        xticklabels=_sparse_ms_labels(rel_b),
        yticklabels=_sparse_ms_labels(rel_a),
        cbar_kws={
            'label': 'mean Pearson r (mean over imgs)',
            'shrink': 0.55,
            'aspect': 28,
        },
    )
    row_labels = pivot_plot.index.to_numpy(dtype=int)
    col_labels = pivot_plot.columns.to_numpy(dtype=int)
    row_pos = {int(v): i for i, v in enumerate(row_labels)}
    col_pos = {int(v): j for j, v in enumerate(col_labels)}
    common_starts = sorted(set(row_pos.keys()) & set(col_pos.keys()))
    if common_starts:
        xs = [col_pos[v] + 0.5 for v in common_starts]
        ys = [row_pos[v] + 0.5 for v in common_starts]
        ax.plot(
            xs,
            ys,
            color=(0.92, 0.92, 0.92),
            linewidth=1.4,
            linestyle='-',
            zorder=5,
            clip_on=True,
        )
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'Time window B start (ms)\n(win_len={wl}; shuffle_level_B={int(shuffle_b)})')
    ax.set_ylabel(f'Time window A start (ms)\n(win_len={wl}; shuffle_level_A={int(shuffle_a)})')
    img_rng = f'{int(img_indices.min())}-{int(img_indices.max())}'
    ax.set_title(
        f'{brain_area} | {cell_group} (n={n_cells} cells)\n'
        f'Mean over Img_Index ∈ [{img_rng}] (n_imgs={n_imgs_used}) | '
        f'shuffle A={int(shuffle_a)} vs B={int(shuffle_b)}'
    )
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax



fig_avg, ax_avg = plot_shuffle_timecorr_heatmap_mean_imgs(
    all_img_corr,
    avg_img_indices,
    plot_shuffle_a,
    plot_shuffle_b,
    cell_group=plot_cell_group,
    t_ms_offset=t_ms0,
    tick_ms_step=20,
    plot_t_end_ms=220,
    save_path=None,
    vmax=vmax,
)
plt.show()



