
import os
import struct
import h5py
import pickle
import pandas as pd


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

def Join(path1,path2):

    joined_path = os.path.join(path1,path2)
    return joined_path

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