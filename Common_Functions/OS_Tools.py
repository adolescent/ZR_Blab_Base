
import os



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
