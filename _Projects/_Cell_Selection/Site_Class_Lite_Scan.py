'''
One-time / occasional scan of Metamers site-class joblibs.

Builds a lite index (stimset, brain_areas, site_name) from filenames — no full
SRS load unless the filename cannot be parsed. Summary scripts read this index
to skip irrelevant sites before loading heavy joblib payloads.

Run after adding new site-class files, or when summary reports a stale index.
'''

#%% paths

import OS_Tools as ot
from Py_Structure.Site_Class_Lite import (
    DEFAULT_INDEX_PATH,
    LITE_VERSION,
    refresh_site_class_index,
    index_summary,
    load_site_class_index,
)

site_class_mlmsb = r'E:\#Preprocessed_Data\SiteClass\Metamers\ML_MSB'
site_class_alasb = r'E:\#Preprocessed_Data\SiteClass\Metamers\AL_ASB'
site_class_alo = r'E:\#Preprocessed_Data\SiteClass\Metamers\ALO'

site_class_alo = r'E:\#Preprocessed_Data\SiteClass\Metamers\ALO'

INDEX_PATH = DEFAULT_INDEX_PATH

SITE_ROOTS = {
    'ML_MSB': site_class_mlmsb,
    'AL_ASB': site_class_alasb,
    'ALO': site_class_alo,
}


#%% scan (incremental: only re-parse changed files)

index_df = refresh_site_class_index(SITE_ROOTS, INDEX_PATH, show_progress=True)

print(f'\nIndex saved -> {INDEX_PATH}')
print(f'CSV copy    -> {INDEX_PATH.replace(".joblib", ".csv")}')
print(f'Total sites: {len(index_df)}')
print(f'  filename parse: {(index_df["parse_method"] == "filename").sum()}')
print(f'  joblib fallback: {(index_df["parse_method"] == "joblib").sum()}')
print(f'  errors: {(index_df["parse_method"] == "error").sum()}')

print('\nStimset counts (all):')
print(index_summary(index_df).to_string(index=False))

print('\nMetamer_NSD sites:')
print(index_summary(index_df, 'Metamer_NSD').to_string(index=False))

print('\nMetamer_1k sites (7 stimsets):')
print(index_summary(index_df, 'Metamer_1k').to_string(index=False))
