'''
This script will generate "silct_info.tsv" in standard format, for FOBCS GUI usage.
Stimulus set are located at "Z:\Monkey\Stimuli\LXY\silct_npx_1416"
1200(texture-boulder-filled)+72(24body-24face-24object)*3

'''

#%%
import os
from tqdm import tqdm 

with open('metamer_info.tsv', 'w') as file:
    # Write heads
    file.write("Index\tFileName\tCategory\tFOB\n")
    # Write stimulus index. For index1-1200, stim in sequence 'Texture-Boulder-Filled'

    for i in range(1,1001):
        c_filename = str(10000+i)[1:]+'.png'
        mod = i%200
        if mod == 0:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tP4C1_Inani\n") # tail fix 
        elif mod <= 20:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tRaw_Ani\n")
        elif mod <= 40:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tRaw_Inani\n")
        elif mod<=60:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tP4C4_Ani\n")
        elif mod<=80:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tP4C4_Inani\n")
        elif mod<=100:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tP4C3_Ani\n")
        elif mod<=120:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tP4C3_Inani\n")
        elif mod<=140:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tP4C2_Ani\n")
        elif mod<=160:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tP4C2_Inani\n")
        elif mod<=180:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tP4C1_Ani\n")
        elif mod<=200:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tP4C1_Inani\n")



print("Data written to output.txt with tabs.")
