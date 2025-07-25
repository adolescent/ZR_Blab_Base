'''
Overall method for orientation detection.
1. Cut graphs into small pieces
2. Detect pieces with boulders in it.
3. Replace each piece with overall info
4. Fill no-boulder piece with random orientation bars.

'''
#%% basic infos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, color
from Stimuli_Function.Graph_Trans import *
import cv2
import random


# image = io.imread(r'D:\#Data\#stimuli\silct\silct_npx_1416\0011.jpg')
image = io.imread(r'D:\#Data\#stimuli\FOB96\FOB2023short\033.png')
gray_image = color.rgb2gray(image)*255
blurred = cv2.GaussianBlur(gray_image, (11,11),5)
boulders = Boulder_Canny((blurred).astype('u1'),20,25)
plt.imshow(boulders)
image = boulders


grid_size = 20 # size of grid for img cutting. This will cut grids into several small pieces.
bar_length = 8 # size of plotted bar, must smaller than half grid.
lw = 2 # width of bar lines.
ang_tick=1
img_min_pix = 15 # at least 5 pix below value 
on_thres = 0.5 # used for 
bk_phase_rand = True

#### below are calculation parts
if len(image.shape) == 3:
    gray_image = color.rgb2gray(image)
else:
    gray_image = image
height,width = gray_image.shape
height_win = height//grid_size
width_win = width//grid_size

## below are bar rand infos.
angles_pool = [i * ang_tick for i in range(int(360//ang_tick))] # pools for angle random.
bias_pool = np.arange(-grid_size//2,grid_size//2)

#%%#################### STEP1,graph cut and orientation calculation #################
grid_infos = pd.DataFrame(0.0,columns=['X','Y','Orien','Orien_Ticks','Pixnum','X_adj','Y_adj'],index=range(height_win*width_win))

# cycle each graph for 
counter=0
for i in range(height_win):
    for j in range(width_win):
        # center_x = (i * grid_size) + (grid_size // 2)
        # center_y = (j * grid_size) + (grid_size // 2)
        c_piece = gray_image[i*grid_size:(i+1)*grid_size,j*grid_size:(j+1)*grid_size]
        c_piece_rev = 1-c_piece
        # check if there are boulders inside the graph. Ignore if not ok.
        pix_num = (c_piece_rev.flatten()>on_thres).sum()
        if pix_num>img_min_pix:
            # print(f'{i},{j}')
            grid_mag = FFT_2D(c_piece_rev,mag_thres=100)
            c_orien,_ = Spectrum_Angle_Estimation(grid_mag)
            c_orien+=90
            c_orien_ticks = (c_orien//ang_tick)*ang_tick
            x_cen,y_cen=Img_Center(c_piece_rev)
            x_adj = x_cen-grid_size/2
            y_adj = y_cen-grid_size/2

        else:
            c_orien = -1
            c_orien_ticks = random.choice(angles_pool)
            if bk_phase_rand == True:
                x_adj=random.choice(bias_pool)
                y_adj=random.choice(bias_pool)
            else:
                x_adj=0
                y_adj=0

        grid_infos.loc[counter] = [j,i,c_orien,c_orien_ticks,pix_num,x_adj,y_adj]
        counter += 1


#%% #################### STEP2,boulder gragh generation #################
bar_matrix = np.zeros((height,width), dtype=np.uint8)
rand_fill = False


for i in range(len(grid_infos)):
    c_loc = grid_infos.loc[i,:]
    center_x = int((c_loc['X'] * grid_size) + (grid_size / 2)+c_loc['X_adj'])
    center_y = int((c_loc['Y'] * grid_size) + (grid_size / 2)+c_loc['Y_adj'])
    # center_x = int((c_loc['X'] * grid_size) + (grid_size / 2))
    # center_y = int((c_loc['Y'] * grid_size) + (grid_size / 2))
    # Random angle
    c_angle = c_loc['Orien']
    if c_angle==-1 and rand_fill ==False: # skip non-boulder.
        continue
    angle = c_loc['Orien_Ticks']

    # Calculate the end coordinates of the bar
    rad_angle = np.deg2rad(angle)
    start_x = int(center_x - (bar_length / 2) * np.cos(rad_angle))
    start_y = int(center_y - (bar_length / 2) * np.sin(rad_angle))
    end_x = int(center_x + (bar_length / 2) * np.cos(rad_angle))
    end_y = int(center_y + (bar_length / 2) * np.sin(rad_angle))
    
    # Draw the bar (line) on the matrix
    cv2.line(bar_matrix, (start_x, start_y), (end_x, end_y), 255, 2)

fig,ax = plt.subplots(ncols=2,nrows=1,figsize=(12,6))
ax[0].imshow(bar_matrix,cmap='gray')
ax[1].imshow(gray_image,cmap='gray')

# %%
