
'''
Codes for standarization operation to get Metamer_NSD stimuli from each site class.

Five brain areas (ML, MSB, AL, ASB, ALO): select cells, export averaged responses,
PSTH, trial-level raw data, FOB tuning, and summary figures.
ALO selects all neurons passing noise ceiling (no FOB preference filter); other areas unchanged.

Incremental export: only reprocess sites whose joblib mtime changed (or are new).
Unchanged sites reuse saved arrays without reloading joblibs.

Stimulus layout (Metamer_NSD stimset, 2216 total in recording):
  - FOB72 tuning block: indices 0–71 (exported separately via FOB pass)
  - FOB repeat block: indices 72–215 (not in response export)
  - Response block (exported): indices 216–2215 (2000 images)
      [0:1000]    Metamer 1000 (stim 216–1215)
      [1000:2000] NSD 1000 (stim 1216–2215)
'''


#%% paths and imports

from Py_Structure.Info_Files.InfoLoader import Select_Cell_Info, Load_Info
from Py_Structure.Struct_Funcs import Single_Recording_Site
from Py_Structure.Site_Class_Lite import (
    DEFAULT_INDEX_PATH,
    LITE_VERSION,
    load_site_class_index,
    refresh_site_class_index,
    sites_for_area,
)
import OS_Tools as ot
import joblib as JL
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import gc
import warnings


site_class_alasb = r'E:\#Preprocessed_Data\SiteClass\Metamers\AL_ASB'
site_class_mlmsb = r'E:\#Preprocessed_Data\SiteClass\Metamers\ML_MSB'
site_class_alo = r'E:\#Preprocessed_Data\SiteClass\Metamers\ALO'

savepath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Metamer_NSD_2k'
SITE_INDEX_PATH = DEFAULT_INDEX_PATH
RUN_LITE_SCAN = True   # True: refresh index before export (run Site_Class_Lite_Scan.py instead)


RUN_SITE_REFRESH = False  # set True to run once, then set back to False
select_mod = 'Metamer_NSD'
N_IMG_METAMER = 1000
N_IMG_NSD = 1000
N_IMG = N_IMG_METAMER + N_IMG_NSD   # 2000; matches InfoLoader np.arange(216, 2216)
SLICE_METAMER = slice(0, N_IMG_METAMER)
SLICE_NSD = slice(N_IMG_METAMER, N_IMG)
N_TIME = 450
TIME_SLICE = slice(150, 320)   # 50–219 ms, 1 ms bins, onset=300
T_MS = np.arange(-100, -100 + N_TIME)
CEILING_THRES = 0.3
DP_THRES = 0.5
MAX_REPEAT = 20
TRIAL_BIN_MS = 5
N_TIME_BIN = N_TIME // TRIAL_BIN_MS   # 450 -> 90
FOB_LENGTHS = {'STI150': 150, 'Wordloc': 180, 'FOB72': 72}
N_FOB_MAX = 150
FOB_TIME_SLICE = slice(160, 320)   # same window as Stim_Cell_Rearrange
BRAIN_AREAS = ['ML', 'MSB', 'AL', 'ASB', 'ALO']
AREA_PREFER = {
    'ML': 'Face', 'AL': 'Face', 'MSB': 'Body', 'ASB': 'Body',
    'ALO': 'all',
}
AREA_FOLDER = {
    'ML': site_class_mlmsb, 'MSB': site_class_mlmsb,
    'AL': site_class_alasb, 'ASB': site_class_alasb,
    'ALO': site_class_alo,
}


#%% helpers

_SUBJ_CANON = {
    'md': 'MD', 'maodan': 'MD',
    'jj': 'JJ', 'jianjian': 'JJ',
    'facai': 'FC', 'fc': 'FC',
    'zhuangzhuang': 'ZZ', 'zz': 'ZZ',
    'fld': 'FLD', 'faladi': 'FLD',
    'dique': 'DQ', 'dq': 'DQ',
}


def normalize_subject(name):
    return _SUBJ_CANON.get(str(name).lower(), name)


def fix_mf_filename(path):
    """Rename joblib path if filename contains MF instead of ML."""
    folder, fname = os.path.split(path)
    new_fname = fname.replace('_MSB_MF_', '_MSB_ML_').replace('_MF_', '_ML_')
    if new_fname == fname:
        return path
    new_path = ot.Join(folder, new_fname)
    if os.path.exists(path) and not os.path.exists(new_path):
        os.rename(path, new_path)
    return new_path if os.path.exists(new_path) else path


def refresh_site_metrics(SRS):
    """Recompute noise ceiling, FOB tuning, and update Site_Info (no gn_dic needed)."""
    SRS.brain_areas = ['ML' if a == 'MF' else a for a in SRS.brain_areas]

    parts = SRS.site_name.split('_')
    if len(parts) > 1:
        parts[1] = normalize_subject(parts[1])
        SRS.site_name = '_'.join(parts)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='.*constant.*')
        SRS.Noise_Ceiling(method='all')
        SRS.raw_redplot = SRS.avr_psth[:, :, SRS.used_on].sum(-1)
        SRS.FOB_Tuning_Calculator()

    SRS.Site_Info['Ceiling_Index'] = SRS.ceiling_index
    SRS.Site_Info['Brain_Area'] = [SRS.brain_areas for _ in range(SRS.cellnum)]

    if len(SRS.Cell_FOB_DPrimes) != 0:
        best_dps, best_cats = [], []
        for i in range(SRS.cellnum):
            c_pref = SRS.Cell_FOB_DPrimes[SRS.Cell_FOB_DPrimes.Cell == i]
            best_dps.append(c_pref.D_Prime.max())
            best_cats.append(c_pref.loc[c_pref.D_Prime.idxmax(), 'Category'])
        SRS.Site_Info['Best_D_Prime'] = best_dps
        SRS.Site_Info['Best_Prefer'] = best_cats
    return SRS


def select_cells(SRS, prefer, ceiling=CEILING_THRES, dp_thres=DP_THRES):
    """Select cells; prefer may be a single category or a tuple (union)."""
    if isinstance(prefer, (list, tuple)):
        parts = []
        for p in prefer:
            s, _ = SRS.Cell_Selection(ceiling=ceiling, prefer=p, dp_thres=dp_thres)
            if len(s):
                parts.append(s)
        if not parts:
            return np.array([], dtype=int)
        return np.unique(np.concatenate(parts))
    selected, _ = SRS.Cell_Selection(ceiling=ceiling, prefer=prefer, dp_thres=dp_thres)
    return selected


def site_mtime(path):
    return os.path.getmtime(path)


def entry_needs_refresh(path, entry):
    saved_mtime = entry.get('mtime')
    if saved_mtime is None:
        return False
    try:
        return saved_mtime != site_mtime(path)
    except OSError:
        return True


def classify_site_changes(current_sites, saved_manifest):
    """Return new, changed, removed paths and a lookup for unchanged entries."""
    saved_by_path = {e['path']: e for e in (saved_manifest or [])}
    current_set = set(current_sites)

    new_sites = [p for p in current_sites if p not in saved_by_path]
    removed_sites = [p for p in saved_by_path if p not in current_set]
    changed_sites, unchanged_entries = [], {}
    for p in current_sites:
        if p not in saved_by_path:
            continue
        entry = saved_by_path[p]
        if entry_needs_refresh(p, entry):
            changed_sites.append(p)
        else:
            unchanged_entries[p] = entry

    return new_sites, changed_sites, removed_sites, unchanged_entries


def load_existing_area_export(out_dir):
    manifest_path = ot.Join(out_dir, 'site_manifest.joblib')
    if not os.path.exists(manifest_path):
        return None, None, None, None, None

    manifest = JL.load(manifest_path)
    avr_path = ot.Join(out_dir, 'avr_rsp.npy')
    psth_path = ot.Join(out_dir, 'psth.npy')
    trials_path = ot.Join(out_dir, 'trials_raw.npy')
    if not (os.path.exists(avr_path) and os.path.exists(psth_path)):
        return manifest, None, None, None, None

    prev_n_repeat = None
    meta_path = ot.Join(out_dir, 'trials_raw_meta.npz')
    info_jl = ot.Join(out_dir, 'cell_site_info.joblib')
    if os.path.exists(meta_path):
        meta = np.load(meta_path)
        prev_n_repeat = min(MAX_REPEAT, int(np.max(meta['n_repeat_valid'])))
    elif os.path.exists(info_jl):
        prev_n_repeat = min(MAX_REPEAT, int(JL.load(info_jl)['n_repeat'].max()))

    return (
        manifest,
        np.load(avr_path),
        np.load(psth_path),
        trials_path if os.path.exists(trials_path) else None,
        prev_n_repeat,
    )


def process_single_site(c_site, prefer, stim_infos):
    """Load one site joblib and return arrays + metadata, or None if no cells."""
    try:
        SRS = JL.load(c_site)
    except Exception as e:
        print(f'Load failed {c_site}: {e}')
        return None

    data_ids = np.asarray(stim_infos[SRS.stimset]['Data'])
    if len(data_ids) != N_IMG:
        print(f'Warning {SRS.site_name}: expected {N_IMG} ids, got {len(data_ids)}')

    selected = select_cells(SRS, prefer)
    if len(selected) == 0:
        del SRS
        return None

    raw = SRS.raw_psth[selected][:, :, data_ids, :]
    n_cell, n_repeat, _, _ = raw.shape

    chunk = {
        'path': c_site,
        'mtime': site_mtime(c_site),
        'selected': selected.copy(),
        'data_ids': data_ids,
        'n_cell': n_cell,
        'n_repeat': n_repeat,
        'avr': raw[:, :, :, TIME_SLICE].sum(-1).mean(1).astype(np.float32),
        'psth': raw.mean(1).astype(np.float32),
        'info_rows': _info_rows_for_site(SRS, selected, n_repeat),
    }
    del SRS, raw
    gc.collect()
    return chunk


def _info_rows_for_site(SRS, selected, n_repeat):
    date, subject = parse_site_meta(SRS.site_name)
    dp_face, dp_body = cell_dp_lookup(SRS, selected)
    ceiling = SRS.ceiling_index[selected]
    rows = []
    for local_i in range(len(selected)):
        rows.append({
            'site_name': SRS.site_name,
            'date': date,
            'subject': subject,
            'local_cell_idx': int(selected[local_i]),
            'stimset': SRS.stimset,
            'dprime_face': float(dp_face[local_i]),
            'dprime_body': float(dp_body[local_i]),
            'ceiling_index': float(ceiling[local_i]),
            'n_repeat': int(n_repeat),
        })
    return rows


def _prefer_label(prefer):
    return '+'.join(prefer) if isinstance(prefer, (list, tuple)) else prefer


def match_area(SRS, area):
    areas = getattr(SRS, 'brain_areas', [])
    if area == 'ML':
        return 'ML' in areas or 'MF' in areas
    return area in areas


def parse_site_meta(site_name):
    parts = site_name.split('_')
    date = parts[0] if len(parts) > 0 else ''
    subject = normalize_subject(parts[1]) if len(parts) > 1 else ''
    return date, subject


def cell_dp_lookup(SRS, cell_ids):
    """Face/Body D' for selected cells; NaN if missing."""
    cell_ids = np.asarray(cell_ids)
    if len(SRS.Cell_FOB_DPrimes) == 0:
        return np.full(len(cell_ids), np.nan), np.full(len(cell_ids), np.nan)
    pivot = SRS.Cell_FOB_DPrimes.pivot(index='Cell', columns='Category', values='D_Prime')
    sub = pivot.reindex(cell_ids)
    face = sub['Face'].to_numpy() if 'Face' in sub.columns else np.full(len(cell_ids), np.nan)
    body = sub['Body'].to_numpy() if 'Body' in sub.columns else np.full(len(cell_ids), np.nan)
    return face, body


def bin_trials_psth(raw, bin_ms=TRIAL_BIN_MS):
    """(n_cell, n_repeat, n_img, n_time) spike counts -> sum within bin_ms bins."""
    n_c, n_r, n_img, n_t = raw.shape
    n_t_use = n_t // bin_ms * bin_ms
    return (raw[..., :n_t_use]
            .reshape(n_c, n_r, n_img, -1, bin_ms)
            .sum(-1, dtype=np.float32))


def fill_trials_memmap(site_chunks, n_cell_total, n_repeat_max, out_dir, area_label='',
                       prev_trials_path=None, reuse_paths=None, prev_n_cell=None, prev_n_repeat=None):
    """Stream each site into memmap; reuse prior trials for unchanged sites."""
    reuse_paths = set(reuse_paths or [])
    trials_path = ot.Join(out_dir, 'trials_raw.npy')
    binned_buf = ot.Join(out_dir, '_trials_binned5ms_buf.npy')
    trials_mm = np.memmap(
        trials_path, dtype=np.float32, mode='w+',
        shape=(n_cell_total, n_repeat_max, N_IMG, N_TIME),
    )
    trials_bin_mm = np.memmap(
        binned_buf, dtype=np.float32, mode='w+',
        shape=(n_cell_total, n_repeat_max, N_IMG, N_TIME_BIN),
    )
    trials_rsp = np.full((n_cell_total, n_repeat_max, N_IMG), np.nan, dtype=np.float32)

    prev_mm = None
    if prev_trials_path and os.path.exists(prev_trials_path) and reuse_paths and prev_n_cell and prev_n_repeat:
        prev_mm = np.memmap(
            prev_trials_path, dtype=np.float32, mode='r',
            shape=(prev_n_cell, prev_n_repeat, N_IMG, N_TIME),
        )

    for chunk in tqdm(site_chunks, desc=f'{area_label} write trials'):
        off, n_c = chunk['offset'], chunk['n_cell']
        n_r = min(chunk['n_repeat'], n_repeat_max)

        if chunk['path'] in reuse_paths and prev_mm is not None and 'prev_offset' in chunk:
            old_off = int(chunk['prev_offset'])
            trials_mm[off:off + n_c, :n_r] = prev_mm[old_off:old_off + n_c, :n_r]
        else:
            SRS = JL.load(chunk['path'])
            raw = SRS.raw_psth[chunk['selected']][:, :, chunk['data_ids'], :].astype(np.float32)
            trials_mm[off:off + n_c, :n_r] = raw[:, :n_r]
            del SRS, raw

        trials_bin_mm[off:off + n_c, :n_r] = bin_trials_psth(trials_mm[off:off + n_c, :n_r])
        trials_rsp[off:off + n_c, :n_r] = trials_mm[off:off + n_c, :n_r, :, TIME_SLICE].sum(-1)
        if n_r < n_repeat_max:
            trials_mm[off:off + n_c, n_r:] = np.nan
            trials_bin_mm[off:off + n_c, n_r:] = np.nan
            trials_rsp[off:off + n_c, n_r:] = np.nan

        if chunk['path'] not in reuse_paths:
            gc.collect()

    if prev_mm is not None:
        del prev_mm

    trials_mm.flush()
    trials_bin_mm.flush()
    del trials_mm, trials_bin_mm
    return trials_rsp, binned_buf


def bin_psth_hz(psth_2d, bin_ms=5):
    """(N_cell, N_time) spike counts -> Hz, binned to bin_ms."""
    n_cell, n_t = psth_2d.shape
    n_t_use = n_t // bin_ms * bin_ms
    t_plot = T_MS[:n_t_use].reshape(-1, bin_ms).mean(1)
    fr = (psth_2d[:, :n_t_use]
          .reshape(n_cell, -1, bin_ms).mean(-1) * 1000)
    return fr, t_plot


def plot_heatmap(avr_rsp, out_path, n_img=N_IMG):
    x = np.asarray(avr_rsp, dtype=np.float64)
    row_mean = x.mean(1, keepdims=True)
    row_std = x.std(1, keepdims=True) + 1e-8
    x_norm = (x - row_mean) / row_std

    fig_h = max(4, min(20, x_norm.shape[0] * 0.02))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    sns.heatmap(
        x_norm, ax=ax, cmap='RdBu_r', center=0,
        vmin=-3, vmax=3, xticklabels=False, yticklabels=False,
        cbar_kws={'label': 'z-score (per neuron)'},
    )
    ax.set_xlabel(f'Stimulus index (0–{n_img - 1})')
    ax.set_ylabel('Neuron')
    ax.set_title(
        f'Normalized response — {x_norm.shape[0]} cells × {n_img} images '
        f'(metamer {N_IMG_METAMER} + NSD {N_IMG_NSD})',
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_raster_first40(psth, out_path, n_img=40, bin_ms=5):
    """Raster of mean PSTH over first n_img images, 5 ms bins, green 50–220 ms window."""
    pop = np.asarray(psth[:, :n_img, :], dtype=np.float64).mean(1)  # (N_cell, 450)
    raster, t_plot = bin_psth_hz(pop, bin_ms=bin_ms)
    raster_norm = raster / np.maximum(raster.max(1, keepdims=True), 1e-6)

    fig_h = max(4, min(20, raster_norm.shape[0] * 0.02))
    fig, ax = plt.subplots(figsize=(6, fig_h))
    ax.imshow(
        raster_norm, aspect='auto', origin='lower', interpolation='nearest',
        extent=(t_plot[0], t_plot[-1] + bin_ms, -0.5, raster_norm.shape[0] - 0.5),
        cmap='Greys', vmin=0, vmax=1,
    )
    ax.axvspan(50, 220, color='lightgreen', alpha=0.2, zorder=0)
    ax.axvline(0, color='cyan', ls='--', lw=1)
    ax.set_xlim(-100, 350)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')
    ax.set_title(f'Avg PSTH — first {n_img} images ({bin_ms} ms bin)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def pad_fob_axis(arr, n_fob_max=N_FOB_MAX):
    """Pad last axis to n_fob_max with NaN for stacked storage; native width in fob_valid_len."""
    arr = np.asarray(arr, dtype=np.float32)
    n_valid = int(arr.shape[-1])
    if n_valid >= n_fob_max:
        return arr[..., :n_fob_max], n_valid
    out = np.full(arr.shape[:-1] + (n_fob_max,), np.nan, dtype=np.float32)
    out[..., :n_valid] = arr
    return out, n_valid


def _fob_redplot_from_stim(stim_rsp, fob_ids, fob_style):
    """(n_cell, [n_repeat,] n_stim) -> native (n_cell, [n_repeat,] fob_len); average duplicate FOB blocks."""
    fob_len = FOB_LENGTHS[fob_style]
    n_repeat_fob = len(fob_ids) // fob_len
    arr = stim_rsp[..., fob_ids]
    if n_repeat_fob > 1:
        lead = arr.shape[:-1]
        arr = arr.reshape(*lead, n_repeat_fob, fob_len).mean(axis=-2)
    return arr


def extract_fob_avr(avr_psth, fob_ids, fob_style, time_slice=FOB_TIME_SLICE):
    """Average FOB — same logic as Stim_Cell_Rearrange (avr_psth, repeat-averaged)."""
    redplot = avr_psth[:, :, time_slice].sum(-1).astype(np.float32)
    return _fob_redplot_from_stim(redplot, fob_ids, fob_style)


def extract_fob_by_trial(raw_psth, fob_ids, fob_style, time_slice=FOB_TIME_SLICE):
    """Trial-level FOB from raw_psth (n_cell, n_repeat, n_stim, n_time)."""
    trial_red = raw_psth[:, :, :, time_slice].sum(-1).astype(np.float32)
    return _fob_redplot_from_stim(trial_red, fob_ids, fob_style)


def plot_fob_heatmap(fob_avr, fob_valid_len, out_path, n_fob_max=N_FOB_MAX):
    """Per-neuron z-score heatmap; only valid FOB columns shown (72 or 150), rest masked."""
    fob_valid_len = np.asarray(fob_valid_len, dtype=np.int16)
    n_cols = int(fob_valid_len.max()) if len(fob_valid_len) else n_fob_max
    n_cols = min(n_cols, n_fob_max)

    x = np.asarray(fob_avr[:, :n_cols], dtype=np.float64)
    x_norm = np.full_like(x, np.nan)
    for i in range(x.shape[0]):
        n_v = int(fob_valid_len[i])
        if n_v <= 0:
            continue
        row = x[i, :n_v]
        std = row.std()
        if std < 1e-8:
            x_norm[i, :n_v] = 0.0
        else:
            x_norm[i, :n_v] = (row - row.mean()) / std

    mask = np.ones_like(x_norm, dtype=bool)
    for i, n_v in enumerate(fob_valid_len):
        if n_v > 0:
            mask[i, :min(int(n_v), n_cols)] = False

    fig_h = max(4, min(20, x_norm.shape[0] * 0.02))
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 0.06), fig_h))
    sns.heatmap(
        x_norm, ax=ax, mask=mask, cmap='RdBu_r', center=0,
        vmin=-3, vmax=3, xticklabels=False, yticklabels=False,
        cbar_kws={'label': 'z-score (per neuron, valid FOB only)'},
    )
    ax.set_xlabel(f'FOB stimulus index (0–{n_cols - 1})')
    ax.set_ylabel('Neuron')
    ax.set_title(f'FOB response — {x_norm.shape[0]} cells (FOB72)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def build_site_manifest(site_chunks, info_df):
    """Lightweight per-site index for standalone FOB pass (no full re-export)."""
    manifest = []
    for chunk in site_chunks:
        row = info_df.iloc[chunk['offset']]
        manifest.append({
            'path': chunk['path'],
            'mtime': chunk.get('mtime', site_mtime(chunk['path'])),
            'selected': chunk['selected'],
            'stimset': row['stimset'],
            'site_name': row['site_name'],
            'offset': chunk['offset'],
            'n_cell': chunk['n_cell'],
            'n_repeat': chunk['n_repeat'],
        })
    return manifest


def export_brain_area_incremental(cloc, sites, prefer, stim_infos, save_root=savepath):
    """
    Incrementally export one brain area.
    Reuses saved arrays for unchanged sites; only reloads new/changed joblibs.
    Returns True if data was written, False if skipped (no changes).
    """
    out_dir = ot.Join(save_root, cloc)
    saved_manifest, existing_avr, existing_psth, prev_trials_path, prev_n_repeat = load_existing_area_export(out_dir)
    new_sites, changed_sites, removed_sites, unchanged_entries = classify_site_changes(
        sites, saved_manifest,
    )

    if saved_manifest and not new_sites and not changed_sites and not removed_sites:
        print(f'{cloc}: no site changes, skip export.')
        return False

    if new_sites or changed_sites or removed_sites:
        print(
            f'{cloc}: incremental update — '
            f'new={len(new_sites)}, changed={len(changed_sites)}, removed={len(removed_sites)}, '
            f'unchanged={len(unchanged_entries)}',
        )
    else:
        print(f'{cloc}: full export ({len(sites)} sites)')

    old_info = None
    info_jl = ot.Join(out_dir, 'cell_site_info.joblib')
    if os.path.exists(info_jl):
        old_info = JL.load(info_jl)
    elif os.path.exists(ot.Join(out_dir, 'cell_site_info.csv')):
        old_info = pd.read_csv(ot.Join(out_dir, 'cell_site_info.csv'))

    avr_list, psth_list, site_chunks = [], [], []
    info_rows = []
    n_repeat_valid_all = []
    n_repeat_max = 0
    global_idx = 0
    reuse_paths = set()
    to_process = set(new_sites) | set(changed_sites)

    for c_site in tqdm(sites, total=len(sites), desc=cloc):
        can_reuse = (
            c_site in unchanged_entries
            and existing_avr is not None
            and existing_psth is not None
            and old_info is not None
        )
        if can_reuse:
            entry = unchanged_entries[c_site]
            off_old = int(entry['offset'])
            n_cell = int(entry['n_cell'])
            n_repeat = int(entry['n_repeat'])
            n_repeat_max = max(n_repeat_max, n_repeat)

            avr_list.append(existing_avr[off_old:off_old + n_cell])
            psth_list.append(existing_psth[off_old:off_old + n_cell])
            site_chunks.append({
                'path': c_site,
                'mtime': entry.get('mtime', site_mtime(c_site)),
                'selected': entry['selected'],
                'data_ids': None,
                'offset': global_idx,
                'prev_offset': off_old,
                'n_cell': n_cell,
                'n_repeat': n_repeat,
            })
            reuse_paths.add(c_site)

            sub = old_info.iloc[off_old:off_old + n_cell].copy()
            sub['global_idx'] = np.arange(global_idx, global_idx + n_cell)
            info_rows.extend(sub.to_dict('records'))
            n_repeat_valid_all.extend(sub['n_repeat'].tolist())
            global_idx += n_cell
            continue

        if c_site not in to_process and c_site not in unchanged_entries:
            continue

        chunk = process_single_site(c_site, prefer, stim_infos)
        if chunk is None:
            continue

        n_cell = chunk['n_cell']
        n_repeat = chunk['n_repeat']
        n_repeat_max = max(n_repeat_max, n_repeat)

        avr_list.append(chunk['avr'])
        psth_list.append(chunk['psth'])
        site_chunks.append({
            'path': chunk['path'],
            'mtime': chunk['mtime'],
            'selected': chunk['selected'],
            'data_ids': chunk['data_ids'],
            'offset': global_idx,
            'n_cell': n_cell,
            'n_repeat': n_repeat,
        })
        n_repeat_valid_all.extend([n_repeat] * n_cell)

        for i, row in enumerate(chunk['info_rows']):
            row = dict(row)
            row['global_idx'] = global_idx + i
            info_rows.append(row)

        global_idx += n_cell
        del chunk
        gc.collect()

    if not avr_list:
        print(f'{cloc}: no cells selected, skip.')
        return False

    avr_rsp = np.vstack(avr_list)
    psth = np.vstack(psth_list)
    n_repeat_max = min(MAX_REPEAT, n_repeat_max)
    n_cell_total = avr_rsp.shape[0]
    n_repeat_arr = np.array(n_repeat_valid_all, dtype=np.int16)

    info_df = pd.DataFrame(info_rows)
    ot.Mkdir(out_dir)

    prev_n_cell = len(old_info) if old_info is not None else None
    trials_rsp, binned_buf = fill_trials_memmap(
        site_chunks, n_cell_total, n_repeat_max, out_dir, area_label=cloc,
        prev_trials_path=prev_trials_path, reuse_paths=reuse_paths,
        prev_n_cell=prev_n_cell, prev_n_repeat=prev_n_repeat,
    )

    np.save(ot.Join(out_dir, 'avr_rsp.npy'), avr_rsp)
    np.save(ot.Join(out_dir, 'psth.npy'), psth)
    info_df.to_csv(ot.Join(out_dir, 'cell_site_info.csv'), index=False)
    JL.dump(info_df, ot.Join(out_dir, 'cell_site_info.joblib'), compress=3)

    prefer_label = _prefer_label(prefer)
    np.savez_compressed(
        ot.Join(out_dir, 'trials_raw_meta.npz'),
        n_repeat_valid=n_repeat_arr,
        brain_area=cloc,
        prefer=prefer_label,
        bin_ms=TRIAL_BIN_MS,
    )
    trials_binned = np.memmap(
        binned_buf, dtype=np.float32, mode='r',
        shape=(n_cell_total, n_repeat_max, N_IMG, N_TIME_BIN),
    )
    np.savez(
        ot.Join(out_dir, 'trials_raw_binned5ms.npz'),
        trials=trials_binned,
        n_repeat_valid=n_repeat_arr,
        brain_area=np.array(cloc),
        prefer=np.array(prefer_label),
        bin_ms=np.int16(TRIAL_BIN_MS),
    )
    del trials_binned
    if os.path.exists(binned_buf):
        os.remove(binned_buf)

    np.savez_compressed(
        ot.Join(out_dir, 'trials_rsp.npz'),
        trials_rsp=trials_rsp,
        n_repeat_valid=n_repeat_arr,
        brain_area=cloc,
        prefer=prefer_label,
    )

    plot_heatmap(avr_rsp, ot.Join(out_dir, 'heatmap_2k.png'))
    plot_raster_first40(psth, ot.Join(out_dir, 'raster_first40.png'))

    site_manifest = build_site_manifest(site_chunks, info_df)
    JL.dump(site_manifest, ot.Join(out_dir, 'site_manifest.joblib'), compress=3)

    del trials_rsp, avr_rsp, psth, site_chunks, site_manifest
    gc.collect()

    print(f'{cloc}: saved {n_cell_total} cells -> {out_dir}')
    return True


def export_stim_layout(save_root=savepath):
    """One-time stimulus index map at save_root (shared across brain areas)."""
    stim_infos = Select_Cell_Info(select_mod)
    data_ids = np.asarray(stim_infos['Metamer_NSD']['Data'], dtype=np.int32)
    tsv_info, _, _ = Load_Info('Metamer_NSD')
    if tsv_info is None:
        raise FileNotFoundError('Metamer_NSD.tsv not found in Info_Files')

    stim_set = tsv_info.iloc[data_ids]['Stim_Set'].to_numpy(dtype=object)
    category = tsv_info.iloc[data_ids]['Category'].to_numpy(dtype=object)
    obj_id = tsv_info.iloc[data_ids]['Object'].to_numpy(dtype=np.int32)

    stim_type = np.empty(len(data_ids), dtype=object)
    stim_type[SLICE_METAMER] = 'metamer'
    stim_type[SLICE_NSD] = 'nsd'

    out_path = ot.Join(save_root, 'stim_layout.npz')
    ot.Mkdir(save_root)
    np.savez_compressed(
        out_path,
        data_ids=data_ids,
        stim_index=np.arange(len(data_ids), dtype=np.int32),
        stim_set=stim_set,
        category=category,
        object_id=obj_id,
        stim_type=stim_type,
        slice_metamer=np.array(SLICE_METAMER.indices(N_IMG), dtype=np.int32),
        slice_nsd=np.array(SLICE_NSD.indices(N_IMG), dtype=np.int32),
        n_img=np.int32(N_IMG),
        n_metamer=np.int32(N_IMG_METAMER),
        n_nsd=np.int32(N_IMG_NSD),
    )
    print(f'stim_layout saved -> {out_path}')
    return out_path


def export_fob_for_area(cloc, save_root=savepath, stim_infos=None):
    """
    Standalone FOB export: reads site_manifest.joblib + cell_site_info from save_root/<cloc>/.
    Loads each site joblib once; uses saved cell indices (no Cell_Selection).
    """
    if stim_infos is None:
        stim_infos = Select_Cell_Info(select_mod)

    out_dir = ot.Join(save_root, cloc)
    manifest_path = ot.Join(out_dir, 'site_manifest.joblib')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f'missing {manifest_path} — run main export first')

    info_jl = ot.Join(out_dir, 'cell_site_info.joblib')
    info_df = JL.load(info_jl) if os.path.exists(info_jl) else pd.read_csv(
        ot.Join(out_dir, 'cell_site_info.csv'),
    )
    site_manifest = JL.load(manifest_path)
    n_cell_total = len(info_df)
    n_repeat_max = min(MAX_REPEAT, int(info_df['n_repeat'].max()))

    fob_by_trial = np.full(
        (n_cell_total, n_repeat_max, N_FOB_MAX), np.nan, dtype=np.float32,
    )
    fob_avr = np.full((n_cell_total, N_FOB_MAX), np.nan, dtype=np.float32)
    fob_valid_len = np.zeros(n_cell_total, dtype=np.int16)
    fob_style_arr = np.empty(n_cell_total, dtype=object)

    for entry in tqdm(site_manifest, desc=f'{cloc} FOB'):
        SRS = JL.load(entry['path'])
        stimset = entry['stimset']
        selected = entry['selected']
        off, n_c = entry['offset'], entry['n_cell']
        n_r = min(entry['n_repeat'], n_repeat_max)

        fob_info = stim_infos[stimset]['FOB']
        fob_style = fob_info['style']
        fob_ids = fob_info['id']

        raw = SRS.raw_psth[selected]
        avr_p = SRS.avr_psth[selected]
        fob_tri = extract_fob_by_trial(raw, fob_ids, fob_style)
        fob_mean = extract_fob_avr(avr_p, fob_ids, fob_style)
        fob_tri_pad, n_valid = pad_fob_axis(fob_tri)
        fob_mean_pad, _ = pad_fob_axis(fob_mean)

        fob_by_trial[off:off + n_c, :n_r] = fob_tri_pad[:, :n_r]
        fob_avr[off:off + n_c] = fob_mean_pad
        fob_valid_len[off:off + n_c] = n_valid
        fob_style_arr[off:off + n_c] = fob_style

        del SRS, raw, avr_p, fob_tri, fob_mean, fob_tri_pad, fob_mean_pad
        gc.collect()

    np.savez_compressed(
        ot.Join(out_dir, 'fob_by_trial.npz'),
        fob_by_trial=fob_by_trial,
        n_repeat_valid=info_df['n_repeat'].to_numpy(dtype=np.int16),
        n_fob_max=np.int16(N_FOB_MAX),
        brain_area=cloc,
    )
    np.save(ot.Join(out_dir, 'fob_avr.npy'), fob_avr)
    np.savez_compressed(
        ot.Join(out_dir, 'fob_meta.npz'),
        fob_valid_len=fob_valid_len,
        fob_style=fob_style_arr,
        n_fob_max=np.int16(N_FOB_MAX),
        brain_area=cloc,
    )
    plot_fob_heatmap(fob_avr, fob_valid_len, ot.Join(out_dir, 'heatmap_fob.png'))

    print(f'{cloc}: FOB saved {n_cell_total} cells -> {out_dir}')
    return fob_by_trial, fob_avr


#%% site-class lite index (skip irrelevant joblibs without full load)

_SITE_ROOTS = {
    'ML_MSB': site_class_mlmsb,
    'AL_ASB': site_class_alasb,
    'ALO': site_class_alo,
}
if RUN_LITE_SCAN or not os.path.exists(SITE_INDEX_PATH):
    site_index = refresh_site_class_index(_SITE_ROOTS, SITE_INDEX_PATH, show_progress=True)
else:
    site_index = refresh_site_class_index(_SITE_ROOTS, SITE_INDEX_PATH, show_progress=False)

print(f'site_class_lite v{LITE_VERSION}, columns={list(site_index.columns)}')


#%% =============================================================================
# ONE-TIME ONLY — refresh site_class: MF→ML, redo noise ceiling & FOB tuning
# Skip this cell on routine re-exports.
# =============================================================================



if RUN_SITE_REFRESH:
    warnings.filterwarnings('ignore', message='.*constant.*')
    _nsd_paths = site_index[site_index['stimset'] == 'Metamer_NSD']['path'].tolist()
    for c_site in tqdm(_nsd_paths, desc='refresh Metamer_NSD'):
        try:
            SRS = JL.load(c_site)
            SRS = refresh_site_metrics(SRS)
            save_to = fix_mf_filename(c_site)
            JL.dump(SRS, save_to, compress=7)
            if save_to != c_site and os.path.exists(c_site):
                os.remove(c_site)
        except Exception as e:
            print(f'Failed {c_site}: {e}')
        finally:
            gc.collect()
    print('Site-class refresh done.')


#%% export stimulus layout (shared, run once)

_stim_layout_path = ot.Join(savepath, 'stim_layout.npz')
if not os.path.exists(_stim_layout_path):
    export_stim_layout(savepath)
else:
    print(f'stim_layout exists, skip -> {_stim_layout_path}')


#%% export brain areas (incremental)

stim_infos = Select_Cell_Info(select_mod)
areas_updated = []

for cloc in BRAIN_AREAS:
    data_path = AREA_FOLDER[cloc]
    prefer = AREA_PREFER[cloc]
    sites = sites_for_area(site_index, data_path, select_mod, cloc)
    if not sites:
        print(f'{cloc}: no matching sites in lite index, skip.')
        continue
    print(f'{cloc}: {len(sites)} sites match {select_mod}')
    if export_brain_area_incremental(cloc, sites, prefer, stim_infos, savepath):
        areas_updated.append(cloc)


#%% FOB export (only for areas updated this run)

RUN_FOB_EXPORT = True

if RUN_FOB_EXPORT:
    _stim_infos_fob = Select_Cell_Info(select_mod)
    for _cloc in areas_updated:
        try:
            export_fob_for_area(_cloc, savepath, _stim_infos_fob)
        except FileNotFoundError as _e:
            print(f'{_cloc} FOB skip: {_e}')
        except Exception as _e:
            print(f'{_cloc} FOB failed: {_e}')
    if not areas_updated:
        print('FOB export skipped (no area updates this run).')
