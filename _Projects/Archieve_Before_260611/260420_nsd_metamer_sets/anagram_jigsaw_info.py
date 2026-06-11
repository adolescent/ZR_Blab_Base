'''
Generate info file for Anagram_Jigsaw_v260227.tsv

'''
#%%

import pandas as pd
import OS_Tools as ot
import os
from Py_Structure.Info_Files.InfoLoader import Load_Info

a,_,_ = Load_Info('Metamer1300')

jigsaw_filenames = ot.Get_File_Name(r'Z:\Monkey\Stimuli\ZR\Anagram_Jigsaw_v260227','.jpg')
jigsaw_filenames.sort()
#%%
# 1-150 as fob_sti150

# 151-750 follow ids below
n_pairs = 20
n_styles = 5
n_repeats = 3

fob_categories = (
    ['Face'] * 15
    + ['Body'] * 15
    + ['Object'] * 15
    + ['Scene'] * 15
    + ['Food'] * 15
    + ['Face_g'] * 15
    + ['Body_g'] * 15
    + ['Object_g'] * 15
    + ['Scene_g'] * 15
    + ['Food_g'] * 15
)

categories = []
stim_sets = []
objects = []

# 1-150: follow FOB_STI150 format, one repeat only.
for c_cat in fob_categories:
    categories.append(c_cat)
    stim_sets.append('FOB_STI150')
    objects.append(-1)

# 151-750: AB pair -> style -> object, 200 images per repeat, repeated 3 times.
for _ in range(n_repeats):
    for obj_id in range(1, n_pairs + 1):
        for style_id in range(1, n_styles + 1):
            for c_suffix in ('A', 'B'):
                categories.append(f'Obj{obj_id}_Style{style_id}_{c_suffix}')
                stim_sets.append('Anagra_Jigsaw')
                objects.append(obj_id)

if len(categories) != len(jigsaw_filenames):
    raise ValueError(
        f'Generated {len(categories)} rows, but found {len(jigsaw_filenames)} image files.'
    )

jigsaw_info = pd.DataFrame(
    {
        'FileName': [os.path.basename(c_path) for c_path in jigsaw_filenames],
        'Category': categories,
        'Stim_Set': stim_sets,
        'Object': objects,
    }
)

# Optional save:
# jigsaw_info.to_csv(
#     r'c:\#working_folder\#Codes\ZR_Blab_Base\Py_Structure\Info_Files\Anagram_Jigsaw_v260227.tsv',
#     sep='\t',
#     index=True
# )
