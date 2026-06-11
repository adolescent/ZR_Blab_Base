
'''
This script will generate redplots for all cell.

Different from A14_Redplot.py, this script will Z score the response before generating redplot.

Including FOB redplot and 
'''


#%%


from nt import read
import seaborn as sns
import OS_Tools as ot
from PIL import Image
import numpy as np
from Matrix_Tools import Corr_Matrix
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

