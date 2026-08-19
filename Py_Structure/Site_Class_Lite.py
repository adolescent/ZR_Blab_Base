'''
Fast site-class index: parse stimset / brain_areas from joblib filenames
(with joblib fallback), cache to disk for summary scripts to skip irrelevant sites.
'''

from Py_Structure.Info_Files.InfoLoader import Select_Cell_Info
import OS_Tools as ot
import joblib as JL
import pandas as pd
import numpy as np
import os

LITE_VERSION = 3
BRAIN_AREA_TOKENS = frozenset({'ML', 'MSB', 'AL', 'ASB', 'ALO', 'MF', 'V4'})
DEFAULT_INDEX_PATH = r'E:\#Preprocessed_Data\SiteClass\Metamers\site_class_lite_index.joblib'
INDEX_COLUMNS = (
    'path', 'folder', 'site_name', 'brain_areas', 'stimset',
    'mtime', 'size', 'parse_method', 'error',
)


def _index_csv_path(index_path):
    return os.path.splitext(index_path)[0] + '.csv'


def _clean_index_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().strip('\ufeff\r\n\t') for c in df.columns]
    if 'Unnamed: 0' in df.columns and 'path' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df


def _coerce_to_dataframe(obj):
    """Accept DataFrame / list[dict] / dict saved by older index versions."""
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return _clean_index_columns(obj)
    if isinstance(obj, pd.Series):
        return _clean_index_columns(obj.to_frame().T)
    if isinstance(obj, list):
        if not obj:
            return pd.DataFrame(columns=list(INDEX_COLUMNS))
        return _clean_index_columns(pd.DataFrame(obj))
    if isinstance(obj, dict):
        if not obj:
            return pd.DataFrame(columns=list(INDEX_COLUMNS))
        if all(isinstance(v, dict) for v in obj.values()):
            rows = list(obj.values())
            if rows and 'path' not in rows[0]:
                rows = [{'path': k, **v} for k, v in obj.items()]
            return _clean_index_columns(pd.DataFrame(rows))
    raise TypeError(f'unsupported index type: {type(obj)}')


def _missing_index_columns(df):
    if df is None:
        return list(INDEX_COLUMNS)
    return [c for c in INDEX_COLUMNS if c not in df.columns]


def _read_raw_index(index_path):
    """Load index from joblib and/or csv; prefer the more complete / newer copy."""
    csv_path = _index_csv_path(index_path)
    joblib_df = csv_df = None

    if os.path.exists(index_path):
        try:
            joblib_df = _coerce_to_dataframe(JL.load(index_path))
        except Exception:
            joblib_df = None

    if os.path.exists(csv_path):
        try:
            csv_df = _clean_index_columns(pd.read_csv(csv_path, encoding='utf-8-sig'))
        except Exception:
            csv_df = None

    if joblib_df is None and csv_df is None:
        return pd.DataFrame(columns=list(INDEX_COLUMNS))

    if joblib_df is not None and csv_df is not None:
        j_miss = len(_missing_index_columns(joblib_df))
        c_miss = len(_missing_index_columns(csv_df))
        if c_miss < j_miss:
            return csv_df
        if j_miss < c_miss:
            return joblib_df
        if os.path.exists(index_path):
            j_mtime = os.path.getmtime(index_path)
            c_mtime = os.path.getmtime(csv_path)
            return csv_df if c_mtime >= j_mtime else joblib_df
        return csv_df

    return joblib_df if joblib_df is not None else csv_df


def all_known_stimsets():
    names = set()
    for mod in ('Anagram', 'Doodle', 'Metamer_1k', 'Metamer_NSD'):
        names.update(Select_Cell_Info(mod).keys())
    return sorted(names, key=len, reverse=True)


def parse_joblib_filename(path, known_stimsets=None):
    if known_stimsets is None:
        known_stimsets = all_known_stimsets()

    stem = os.path.splitext(os.path.basename(path))[0]
    stimset = None
    prefix = stem
    for name in known_stimsets:
        suffix = '_' + name
        if stem.endswith(suffix):
            stimset = name
            prefix = stem[:-len(suffix)]
            break
    if stimset is None:
        return None

    parts = prefix.split('_')
    areas = []
    while parts and parts[-1] in BRAIN_AREA_TOKENS:
        areas.insert(0, parts.pop())
    if not areas:
        return None

    return {
        'site_name': '_'.join(parts),
        'brain_areas': areas,
        'stimset': stimset,
    }


def parse_joblib_meta(path, known_stimsets=None):
    meta = parse_joblib_filename(path, known_stimsets=known_stimsets)
    if meta is not None:
        meta['parse_method'] = 'filename'
        return meta

    SRS = JL.load(path)
    areas = ['ML' if a == 'MF' else a for a in getattr(SRS, 'brain_areas', [])]
    meta = {
        'site_name': getattr(SRS, 'site_name', ''),
        'brain_areas': areas,
        'stimset': getattr(SRS, 'stimset', ''),
        'parse_method': 'joblib',
    }
    del SRS
    return meta


def _file_sig(path):
    st = os.stat(path)
    return st.st_mtime, st.st_size


def _row_from_path(path, folder_label, known_stimsets):
    mtime, size = _file_sig(path)
    try:
        meta = parse_joblib_meta(path, known_stimsets=known_stimsets)
    except Exception as exc:
        return {
            'path': path,
            'folder': folder_label,
            'site_name': '',
            'brain_areas': '',
            'stimset': '',
            'mtime': mtime,
            'size': size,
            'parse_method': 'error',
            'error': str(exc),
        }

    return {
        'path': path,
        'folder': folder_label,
        'site_name': meta['site_name'],
        'brain_areas': ','.join(meta['brain_areas']),
        'stimset': meta['stimset'],
        'mtime': mtime,
        'size': size,
        'parse_method': meta['parse_method'],
        'error': '',
    }


def _normalize_index_row(row, known_stimsets=None):
    if known_stimsets is None:
        known_stimsets = all_known_stimsets()

    if isinstance(row, pd.Series):
        row = row.to_dict()
    else:
        row = dict(row)

    out = {col: row.get(col, np.nan) for col in INDEX_COLUMNS}

    pm = out.get('parse_method')
    if pm is None or (isinstance(pm, float) and pd.isna(pm)) or pm == '':
        out['parse_method'] = 'legacy'

    path = out.get('path', '')
    stimset = out.get('stimset', '')
    if (stimset is None or (isinstance(stimset, float) and pd.isna(stimset)) or stimset == '') and path:
        meta = parse_joblib_filename(path, known_stimsets=known_stimsets)
        if meta is not None:
            out['stimset'] = meta['stimset']
            out['brain_areas'] = ','.join(meta['brain_areas'])
            out['site_name'] = meta['site_name']
            out['parse_method'] = 'filename'

    err = out.get('error')
    if err is None or (isinstance(err, float) and pd.isna(err)):
        out['error'] = ''

    return out


def _normalize_index_df(df, known_stimsets=None):
    df = _coerce_to_dataframe(df)
    if df is None or len(df) == 0:
        return pd.DataFrame({c: pd.Series(dtype='object') for c in INDEX_COLUMNS})

    for col in INDEX_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan if col in ('mtime', 'size') else ''

    df = df.reindex(columns=list(INDEX_COLUMNS))

    if known_stimsets is None:
        known_stimsets = all_known_stimsets()

    fixed_rows = [_normalize_index_row(r, known_stimsets=known_stimsets) for r in df.to_dict('records')]
    out = pd.DataFrame(fixed_rows, columns=list(INDEX_COLUMNS))
    for num_col in ('mtime', 'size'):
        out[num_col] = pd.to_numeric(out[num_col], errors='coerce')
    out['parse_method'] = out['parse_method'].fillna('legacy').replace('', 'legacy')
    out['error'] = out['error'].fillna('')
    return out


def scan_site_class_roots(roots, known_stimsets=None, show_progress=True):
    if known_stimsets is None:
        known_stimsets = all_known_stimsets()

    if isinstance(roots, dict):
        root_items = list(roots.items())
    else:
        root_items = [(os.path.basename(r), r) for r in roots]

    rows = []
    iterator = root_items
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(root_items, desc='scan roots')

    for folder_label, root in iterator:
        if not os.path.isdir(root):
            continue
        for path in ot.Get_File_Name(root, '.joblib'):
            rows.append(_row_from_path(path, folder_label, known_stimsets))

    return _normalize_index_df(rows, known_stimsets=known_stimsets)


def refresh_site_class_index(roots, index_path=DEFAULT_INDEX_PATH, show_progress=True):
    if isinstance(roots, dict):
        root_items = list(roots.items())
    else:
        root_items = [(os.path.basename(r), r) for r in roots]

    known_stimsets = all_known_stimsets()
    cached = {}
    if os.path.exists(index_path) or os.path.exists(_index_csv_path(index_path)):
        cached = {
            row['path']: row
            for row in _normalize_index_df(
                _read_raw_index(index_path), known_stimsets=known_stimsets,
            ).to_dict('records')
        }

    new_rows = []
    path_iter = []
    for folder_label, root in root_items:
        if not os.path.isdir(root):
            continue
        for path in ot.Get_File_Name(root, '.joblib'):
            path_iter.append((path, folder_label))

    if show_progress:
        from tqdm import tqdm
        path_iter = tqdm(path_iter, desc='refresh index')

    for path, folder_label in path_iter:
        mtime, size = _file_sig(path)
        old = cached.get(path)
        if old and old.get('mtime') == mtime and old.get('size') == size and old.get('stimset'):
            new_rows.append(old)
            continue
        new_rows.append(_row_from_path(path, folder_label, known_stimsets))

    df = _normalize_index_df(new_rows, known_stimsets=known_stimsets)
    save_site_class_index(df, index_path)
    return df


def save_site_class_index(df, index_path=DEFAULT_INDEX_PATH):
    df = _normalize_index_df(df)
    ot.Mkdir(os.path.dirname(index_path))
    JL.dump(df, index_path, compress=3)
    df.to_csv(_index_csv_path(index_path), index=False, encoding='utf-8-sig')


def load_site_class_index(index_path=DEFAULT_INDEX_PATH, resync=True):
    csv_path = _index_csv_path(index_path)
    if not os.path.exists(index_path) and not os.path.exists(csv_path):
        raise FileNotFoundError(
            f'missing site index: {index_path} (and {csv_path})\n'
            'Run Site_Class_Lite_Scan.py first (or set RUN_LITE_SCAN=True in summary).',
        )

    raw = _read_raw_index(index_path)
    df = _normalize_index_df(raw)

    if resync:
        raw_df = _coerce_to_dataframe(raw) if raw is not None else None
        need_save = (
            not os.path.exists(index_path)
            or _missing_index_columns(raw_df)
            or (os.path.exists(csv_path) and os.path.exists(index_path)
                and os.path.getmtime(csv_path) > os.path.getmtime(index_path))
        )
        if need_save:
            save_site_class_index(df, index_path)

    return df


def _brain_areas_list(brain_areas_str):
    if brain_areas_str is None or (isinstance(brain_areas_str, float) and pd.isna(brain_areas_str)):
        return []
    if isinstance(brain_areas_str, list):
        return brain_areas_str
    return [a.strip() for a in str(brain_areas_str).split(',') if a.strip()]


def row_matches_area(row, area):
    areas = _brain_areas_list(row.get('brain_areas', ''))
    if area == 'ML':
        return 'ML' in areas or 'MF' in areas
    return area in areas


def sites_for_area(index_df, folder_path, select_mod, area):
    index_df = _normalize_index_df(index_df)
    stimsets = set(Select_Cell_Info(select_mod).keys())
    folder_path = os.path.normpath(folder_path)

    mask = index_df['path'].apply(lambda p: os.path.normpath(os.path.dirname(str(p))) == folder_path)
    sub = index_df.loc[mask]
    if sub.empty:
        return []

    sub = sub.loc[sub['stimset'].isin(stimsets)]
    if sub.empty:
        return []

    # Avoid DataFrame.apply(axis=1) on 0-row frames — pandas drops all columns.
    area_mask = sub['brain_areas'].map(lambda ba: row_matches_area({'brain_areas': ba}, area))
    sub = sub.loc[area_mask.fillna(False)]
    if sub.empty:
        return []

    parse_method = sub['parse_method'].fillna('legacy').astype(str)
    sub = sub.loc[parse_method != 'error']
    return sub['path'].tolist()


def index_summary(index_df, select_mod=None):
    df = _normalize_index_df(index_df)
    if select_mod is not None:
        stimsets = set(Select_Cell_Info(select_mod).keys())
        df = df[df['stimset'].isin(stimsets)]
    return df.groupby(['folder', 'stimset']).size().reset_index(name='n_sites')
