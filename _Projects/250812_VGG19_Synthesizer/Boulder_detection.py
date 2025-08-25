'''
This script will try to detect boulder from the graph, generate corresponding 
'''

#%%
import cv2
import numpy as np

# 读取图像
image = cv2.imread(r'D:\#stimuli\Scramble_Global_PNAS_VGG19_50_imgnet\raw\cropped\Animate1_albatross.jpg', cv2.IMREAD_COLOR)  # 替换为你的图像路径

# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 高斯模糊降噪 (内核大小推荐为5x5)
blurred = cv2.GaussianBlur(gray, (13,13),3)

# Canny边缘检测
# 参数说明：
#   blurred: 输入图像
#   50: 低阈值
#   150: 高阈值 (推荐比例为1:2或1:3)
edges = cv2.Canny(blurred,25,50)

# 显示结果
# cv2.imshow('Original Image', image)
cv2.imshow('Canny Edges', edges)

# 保存结果
# cv2.imwrite('canny_edges.jpg', edges)

# 等待按键关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
