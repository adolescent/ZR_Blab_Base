'''
Use this code to generate 1k+STI150x2 analysis infos

'''

#%% OLD-1072 ver
import os
from tqdm import tqdm 

with open('metamer_info_1072_ver.tsv', 'w') as file:
    # Write heads
    file.write("Index\tFileName\tCategory\tFOB\n")
    # Write stimulus index. For index1-1200, stim in sequence 'Texture-Boulder-Filled'

    for i in range(1,73):
        c_filename = str(1000+i)[1:]+'.png'
        if i <25:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tFOB_Body\n")
        elif i<49:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tFOB_Face\n")
        else:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tFOB_Object\n")

    for i in range(73,1073):
        c_filename = str(10000+i-72)[1:]+'.jpg'
        mod = (i-72)%200
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

#%% NEW 1300 VER with STI 300x2

with open('metamer_info_1300_ver.tsv', 'w') as file:
    # Write heads
    file.write("Index\tFileName\tCategory\tFOB\n")

    for i in range(1,1001):
        c_filename = str(10000+i)[1:]+'.jpg'
        mod = (i)%200
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

    for i in range(1001,1301):
        c_filename = str(10000+i)[1:]+'.jpg'
        mod = (i-1000)%150
        if mod == 0:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tSTI_Food_g\n") # tail fix 
        elif mod <= 15:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tSTI_Face\n")
        elif mod <= 30:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tSTI_Body\n")
        elif mod <= 45:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tSTI_Scene\n")
        elif mod <= 60:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tSTI_Object\n")
        elif mod <= 75:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tSTI_Food\n")
        elif mod <= 90:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tSTI_Face_g\n")
        elif mod <= 105:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tSTI_Body_g\n")
        elif mod <= 120:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tSTI_Scene_g\n")
        elif mod <= 135:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tSTI_Object_g\n")
        elif mod <= 150:
            file.write(f"{i}\t{c_filename}\tMetamerP4\tSTI_Food_g\n")