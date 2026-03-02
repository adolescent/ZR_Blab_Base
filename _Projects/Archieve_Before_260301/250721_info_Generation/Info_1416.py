'''
This script will generate "silct_info.tsv" in standard format, for FOBCS GUI usage.
Stimulus set are located at "Z:\Monkey\Stimuli\LXY\silct_npx_1416"
1200(texture-boulder-filled)+72(24body-24face-24object)*3

'''

#%%
import os
from tqdm import tqdm 

with open('silct_info.tsv', 'w') as file:
    # Write heads
    file.write("Index\tFileName\tCategory\tFOB\n")
    # Write stimulus index. For index1-1200, stim in sequence 'Texture-Boulder-Filled'
    for i in tqdm(range(1,1201)):
        c_filename = str(10000+i)[1:]+'.jpg'
        if i%3 == 1:
            file.write(f"{i}\t{c_filename}\tSilct_BigData\tTexture\n")
        elif i%3 == 2:
            file.write(f"{i}\t{c_filename}\tSilct_BigData\tBoulder\n")
        elif i%3 == 0:
            file.write(f"{i}\t{c_filename}\tSilct_BigData\tFilled\n")
    # Then write FOB below.
    for i in tqdm(range(1201,1417)):
        c_filename = str(10000+i)[1:]+'.jpg'
        if (i-1201)%72<24:
            file.write(f"{i}\t{c_filename}\tBody_Face_Object\tBody\n")
        elif (i-1201)%72<48:
            file.write(f"{i}\t{c_filename}\tBody_Face_Object\tFace\n")
        else:
            file.write(f"{i}\t{c_filename}\tBody_Face_Object\tObject\n")


print("Data written to output.txt with tabs.")