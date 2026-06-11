'''
This script will generate "silct_info.tsv" in standard format, for FOBCS GUI usage.
Stimulus set are located at "Z:\Monkey\Stimuli\LXY\silct_npx_1416"
1200(texture-boulder-filled)+72(24body-24face-24object)*3

'''

#%%
import os
from tqdm import tqdm 

with open('sti150_info.tsv', 'w') as file:
    # Write heads
    file.write("Index\tFileName\tCategory\tFOB\n")
    # Write stimulus index. For index1-1200, stim in sequence 'Texture-Boulder-Filled'
    for i in range(1,16):
        c_filename = str(1000+i)+'.jpg'
        file.write(f"{i}\t{c_filename}\tSTI150\tFace\n")
    for i in range(16,31):
        c_filename = str(2000+i-15)+'.jpg'
        file.write(f"{i}\t{c_filename}\tSTI150\tBody\n")
    for i in range(31,46):
        c_filename = str(3000+i-30)+'.jpg'
        file.write(f"{i}\t{c_filename}\tSTI150\tScene\n")
    for i in range(46,61):
        c_filename = str(4000+i-45)+'.jpg'
        file.write(f"{i}\t{c_filename}\tSTI150\tObject\n")
    for i in range(61,76):
        c_filename = str(5000+i-60)+'.jpg'
        file.write(f"{i}\t{c_filename}\tSTI150\tFood\n")
    # gray part
    for i in range(76,91):
        c_filename = 'g_'+str(1000+i-75)+'.jpg'
        file.write(f"{i}\t{c_filename}\tSTI150\tFace_g\n")
    for i in range(91,106):
        c_filename = 'g_'+str(2000+i-15-75)+'.jpg'
        file.write(f"{i}\t{c_filename}\tSTI150\tBody_g\n")
    for i in range(106,121):
        c_filename = 'g_'+str(3000+i-30-75)+'.jpg'
        file.write(f"{i}\t{c_filename}\tSTI150\tScene_g\n")
    for i in range(121,136):
        c_filename = 'g_'+str(4000+i-45-75)+'.jpg'
        file.write(f"{i}\t{c_filename}\tSTI150\tObject_g\n")
    for i in range(136,151):
        c_filename = 'g_'+str(5000+i-60-75)+'.jpg'
        file.write(f"{i}\t{c_filename}\tSTI150\tFood_g\n")



print("Data written to output.tsv with tabs.")