
'''
This script will generate tsv file for 400 stimulus. 
Sequence of stim is : (Raw-44-33-22-11),for each set, in range animate,inanimate,silct,texture. 20 for each.

'''

#%%

import os
from tqdm import tqdm 

with open('scramble.tsv', 'w') as file:
    # Write heads
    file.write("Index\tFileName\tCategory\tFOB\n")
    for i in tqdm(range(1,400)):
        c_filename = str(10000+i)[1:]+'.jpg'
        if i<=21:
            file.write(f"{i}\t{c_filename}\tScramble_BigData\tRaw_Animate\n")
        if i>20 and i<41:
            file.write(f"{i}\t{c_filename}\tScramble_BigData\tRaw_Inanimate\n")
        if i>40 and i<61:
            file.write(f"{i}\t{c_filename}\tScramble_BigData\tRaw_Silct\n")
        if i>60 and i<81:
            file.write(f"{i}\t{c_filename}\tScramble_BigData\tRaw_Texture\n")
        if i>80 and i<101:
            file.write(f"{i}\t{c_filename}\tScramble_BigData\tS44_Animate\n")
        if i>100 and i<121:
            file.write(f"{i}\t{c_filename}\tScramble_BigData\tS44_Inanimate\n")
        if i>120 and i<141:
            file.write(f"{i}\t{c_filename}\tScramble_BigData\tS44_Silct\n")
        if i>140 and i<161:
            file.write(f"{i}\t{c_filename}\tScramble_BigData\tS44_Texture\n")
        if i>160 and i<181:
            file.write(f"{i}\t{c_filename}\tScramble_BigData\tS33_Animate\n")
        if i>180 and i<201:
            file.write(f"{i}\t{c_filename}\tScramble_BigData\tS33_Inanimate\n")
        if i>200 and i<221:
            file.write(f"{i}\t{c_filename}\tScramble_BigData\tS33_Silct\n")
        if i>220 and i<241:
            file.write(f"{i}\t{c_filename}\tScramble_BigData\tS33_Texture\n")
        if i>240 and i<261:
            file.write(f"{i}\t{c_filename}\tScramble_BigData\tS33_Animate\n")
        if i>260 and i<281:
            file.write(f"{i}\t{c_filename}\tScramble_BigData\tS33_Inanimate\n")
        if i>280 and i<301:
            file.write(f"{i}\t{c_filename}\tScramble_BigData\tS33_Silct\n")
        if i>220 and i<241:
            file.write(f"{i}\t{c_filename}\tScramble_BigData\tS33_Texture\n")




print("Data written to output.txt with tabs.")