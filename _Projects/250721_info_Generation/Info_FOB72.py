'''
This script will generate "silct_info.tsv" in standard format, for FOBCS GUI usage.
Stimulus set are located at "Z:\Monkey\Stimuli\LXY\silct_npx_1416"
1200(texture-boulder-filled)+72(24body-24face-24object)*3

'''

#%%
import os
from tqdm import tqdm 

with open('FOB72_info.tsv', 'w') as file:
    # Write heads
    file.write("Index\tFileName\tCategory\tFOB\n")
    # Write stimulus index. For index1-1200, stim in sequence 'Texture-Boulder-Filled'
    for i in range(1,25):
        c_filename = str(1000+i)[1:]+'.png'
        file.write(f"{i}\t{c_filename}\tFOB72\tBody\n")
    for i in range(25,49):
        c_filename = str(1000+i)[1:]+'.png'
        file.write(f"{i}\t{c_filename}\tFOB72\tFace\n")
    for i in range(49,73):
        c_filename = str(1000+i)[1:]+'.png'
        file.write(f"{i}\t{c_filename}\tFOB72\tObject\n")



print("Data written to output.tsv with tabs.")