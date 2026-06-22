#%%
import os
import re
import struct
import h5py
import pickle
from collections import defaultdict, deque

import numpy as np
import pandas as pd

# First argument of Mov(...) in condition log, e.g. Mov(./task_video_adj/tool1sfm_Cycle1.avi,0.000,0.000)
_MOV_PATH_RE = re.compile(r"Mov\s*\(\s*([^,)]+)", re.IGNORECASE)


def Get_File_Name(path,file_type = '.jpg',keyword = ''):
    """
    Get all file names of specific type.

    Parameters
    ----------
    path : (str)
        Root path you want to cycle.
    file_type : (str), optional
        File type you want to get. The default is '.tif'.
    keyword : (str), optional
        Key word you need to screen file. Just leave '' if you need all files.

    Returns
    -------
    Name_Lists : (list)
       Return a list, all file names contained.

    """
    Name_Lists=[]
    for root, dirs, files in os.walk(path):
        for file in files:# walk all files in folder and subfolders.
            if root == path:# We look only files in root folder, subfolder ignored.
                if (os.path.splitext(file)[1] == file_type) and (keyword in file):# we need the file have required extend name and keyword contained.
                    Name_Lists.append(os.path.join(root, file))

    return Name_Lists

def Bin_Unpack(bytes,unpack_bit_num,var_len,type):
    '''
    Unpack a specific length of bits, return cutted bits
    '''
    buffer = str(var_len)+type
    unpacked = struct.unpack(buffer,bytes[:unpack_bit_num])
    rest_bytes = bytes[unpack_bit_num:]

    return rest_bytes,unpacked

def H5_File_Tree(val,pre = ''):
    '''
    Warning, this function will return whole . NOT recommended for whole data structure, check for ONLY the part of data you want!
    '''
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                H5_File_Tree(val, pre+'    ')
            else:
                try:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '└── ' + key + ' (scalar)')
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                H5_File_Tree(val, pre+'│   ')
            else:
                try:
                    print(pre + '├── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '├── ' + key + ' (scalar)')

def Join(*paths):
    """Join two or more path segments. Same idea as os.path.join."""
    if len(paths) < 2:
        raise TypeError('Join() expects at least 2 path arguments')
    return os.path.join(*paths)

def Mkdir(path,mute = False):
    '''
    This function will generate folder at input path. If the folder already exists, then do nothing.
    
    Parameters
    ----------
    path : (str)
        Target path you want to generate folder on.
    mute : (bool),optional
        Message will be ignored if mute is True. Default is False
        
    Returns
    -------
    bool
        Whether new folder is generated.

    '''
    isExists=os.path.exists(path)
    if isExists:
        # 如果目录存在则不创建，并提示目录已存在
        if mute == False:
            print('Folder',path,'already exists!')
        return False
    else:
        os.mkdir(path)
        return True
    

def Save_Variable(save_folder,name,variable,extend_name = '.pkl'):
    """
    Save a variable as binary data.

    Parameters
    ----------
    save_folder : (str)
        Save Path. Only save folder.
    name : (str)
        File name.
    variable : (Any Type)
        Data you want to save.
    extend_name : (str), optional
        Extend name of saved file. The default is '.pkl'.

    Returns
    -------
    bool
        Nothing.

    """
    if os.path.exists(save_folder):
        pass 
    else:
        os.mkdir(save_folder)
    real_save_path = save_folder+r'\\'+name+extend_name
    fw = open(real_save_path,'wb')
    pickle.dump(variable,fw)
    fw.close()
    return True

def Load_Variable(save_folder,file_name=False):
    if file_name == False:
        real_file_path = save_folder
    else:
        real_file_path = save_folder+r'\\'+file_name
    if os.path.exists(real_file_path):
        pickle_off = open(real_file_path,"rb")
        loaded_file = pd.read_pickle(pickle_off)
        pickle_off.close()
    else:
        loaded_file = False

    return loaded_file


def _normalize_tsv_filename(fn):
    """Match ``FileName`` column to basename only (e.g. tool1sfm_Cycle1.avi)."""
    s = str(fn).strip()
    if not s:
        return ''
    return os.path.basename(s.replace("\\", "/"))


def _video_filename_from_condition_txt_line(line):
    """
    Parse one line of ML condition log: extract video file from ``Mov(path,x,y)``.

    ``path`` is often like ``./task_video_adj/tool1sfm_Cycle1.avi``; returns
    basename only so it matches tsv ``FileName``.
    """
    s = str(line).strip()
    if not s or s.startswith("#"):
        return ""
    m = _MOV_PATH_RE.search(s)
    if not m:
        return ""
    path = m.group(1).strip().strip("'\"")
    return os.path.basename(path.replace("\\", "/"))


def Tsv_Txt_Align(txt_path, tsv_info):
    """
    Map each tsv row (design order) to the trial index used in GoodUnit rasters.

    ``txt_path`` should be an ML-style condition log in **actual presentation
    order**. Each trial row is expected to contain ``Mov(<path>, ...)`` where
    ``<path>`` points to the video (e.g. ``./task_video_adj/foo.avi``); the
    basename is matched to ``tsv_info['FileName']`` (e.g. ``foo.avi``). Trial
    indices count only lines from which a ``Mov(...)`` path was parsed (0-based,
    in file top-to-bottom order); blank lines are skipped.

    Returns
    -------
    numpy.ndarray, shape (len(tsv_info),), dtype intp
        ``new_seq[j]`` is the trial index (third axis of ``raw_rasters``) where
        ``tsv_info['FileName'].iloc[j]`` was presented, so
        ``raw_rasters[:, :, new_seq, :]`` aligns rasters to tsv row order.

    Raises
    ------
    ValueError
        If a tsv FileName cannot be matched to the txt list, or queues deplete.
    """
    if tsv_info is None or len(tsv_info) == 0:
        raise ValueError('tsv_info is empty or None.')
    if txt_path is None or (isinstance(txt_path, str) and txt_path.strip() in ('', 'None')):
        raise ValueError('txt_path must be a path to the condition-order text file.')

    df = tsv_info.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if 'FileName' not in df.columns:
        raise ValueError("tsv_info must contain a 'FileName' column.")

    with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
        txt_lines = f.readlines()

    positions_by_label = defaultdict(deque)
    trial_idx = 0
    for line in txt_lines:
        raw = line.rstrip("\n\r")
        if not raw.strip():
            continue
        lab = _video_filename_from_condition_txt_line(raw)
        if not lab:
            continue
        positions_by_label[lab].append(trial_idx)
        trial_idx += 1

    new_seq = np.empty(len(df), dtype=np.intp)
    for j in range(len(df)):
        fn = df['FileName'].iloc[j]
        key = _normalize_tsv_filename(fn)
        q = positions_by_label.get(key)
        if not q:
            raise ValueError(
                f"No txt line matches tsv row {j} FileName={fn!r} (normalized {key!r})."
            )
        new_seq[j] = q.popleft()

    # leftover txt entries -> warn via error if strict one-to-one expected
    leftover = sum(len(q) for q in positions_by_label.values())
    if leftover:
        raise ValueError(
            f'{leftover} txt line(s) were not consumed by tsv_info; check length and labels.'
        )

    return new_seq
# %%
