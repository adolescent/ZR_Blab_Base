'''
This script will try to smooth boulder graph, avoid sudden change on orientation.
'''
#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像并二值化
img = cv2.imread(r'D:\#Data\#stimuli\silct\silct_npx_1416\0998.jpg',0)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# 2. 形态学平滑（可选）
kernel = np.ones((5,5), np.uint8)
smoothed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 3. 提取轮廓并简化
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
simplified_contours = []
for cnt in contours:
    epsilon = 0.005 * cv2.arcLength(cnt, True)  # 调整系数控制简化程度
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    simplified_contours.append(approx)

# 4. 绘制结果
result = np.zeros_like(img)
cv2.drawContours(result, simplified_contours, -1, 255, 2)
# cv2.imwrite('smoothed.png', result)
cv2.imshow('test',result)
cv2.waitKey(5000)
cv2.destroyAllWindows()

#%% curve fit methods
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def smooth_contour_with_spline(contour, num_points=100, smoothness=0.1, periodic=True):
    """
    使用样条曲线平滑轮廓
    
    参数:
    contour: 原始轮廓点 (N, 1, 2)
    num_points: 生成的平滑曲线点数
    smoothness: 平滑系数 (0-1)，值越大越平滑
    periodic: 是否为封闭轮廓
    
    返回:
    smoothed: 平滑后的轮廓点 (num_points, 2)
    """
    # 将轮廓点转换为 (N, 2) 格式
    points = contour.reshape(-1, 2)
    
    # 如果是封闭轮廓，添加首点作为尾点以确保闭合
    if periodic:
        points = np.vstack([points, points[0]])
    
    # 样条曲线拟合
    tck, u = splprep(points.T, u=None, s=len(points)*smoothness, per=periodic)
    
    # 生成均匀分布的新点
    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new = splev(u_new, tck, der=0)
    
    # 组合成点集
    smoothed = np.vstack([x_new, y_new]).T
    
    # 如果是封闭轮廓，移除添加的尾点
    if periodic:
        smoothed = smoothed[:-1]
    
    return smoothed.astype(np.int32)

def process_image(img, output_path, 
                  smoothness=0.1, 
                  epsilon_factor=0.01, 
                  spline_points=200,
                  show_steps=False):
    """
    处理图像：平滑轮廓并保存结果
    
    参数:
    image_path: 输入图像路径
    output_path: 输出图像路径
    smoothness: 曲线平滑度 (0-1)
    epsilon_factor: 轮廓简化系数
    spline_points: 样条曲线点数
    show_steps: 是否显示处理过程
    """
    # 读取图像并二值化
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 形态学平滑
    kernel = np.ones((11, 11), np.uint8)
    smoothed_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 提取轮廓
    contours, _ = cv2.findContours(smoothed_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建结果图像
    result = np.ones_like(img) * 255
    
    # 处理每个轮廓
    all_smoothed_contours = []
    for cnt in contours:
        # 简化轮廓
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 使用样条曲线平滑
        smoothed = smooth_contour_with_spline(approx, num_points=spline_points, 
                                             smoothness=smoothness)
        all_smoothed_contours.append(smoothed.reshape(-1, 1, 2))
        
        # 绘制平滑后的轮廓
        cv2.drawContours(result, [smoothed], 0, 0, 1)
    
    # 保存结果
    # cv2.imwrite(output_path, result)
    
    # 显示处理过程
    if show_steps:
        plt.figure(figsize=(15, 10))
        
        # 原始图像
        plt.subplot(231)
        plt.imshow(img, cmap='gray')
        plt.title("Origin")
        
        # 二值化图像
        plt.subplot(232)
        plt.imshow(binary, cmap='gray')
        plt.title("Binary")
        
        # 形态学平滑后
        plt.subplot(233)
        plt.imshow(smoothed_binary, cmap='gray')
        plt.title("Morpho-Smooth")
        
        # 轮廓简化后
        contour_img = np.ones_like(img) * 255
        cv2.drawContours(contour_img, contours, -1, 0, 1)
        plt.subplot(234)
        plt.imshow(contour_img, cmap='gray')
        plt.title("Countour")
        
        # 曲线拟合后
        plt.subplot(235)
        plt.imshow(result, cmap='gray')
        plt.title("Curvature fit")
        
        # 对比图
        plt.subplot(236)
        plt.imshow(img, cmap='gray')
        for cnt in all_smoothed_contours:
            plt.plot(cnt[:, 0, 0], cnt[:, 0, 1], 'r-', linewidth=2)
        plt.title("Compare")
        
        plt.tight_layout()
        plt.show()
    
    return result
#%% test run part
if __name__ == '__main__':
    # input_image = "input_sketch.png"
    input_image = img
    # input_image = cv2.imread(r'D:\#Data\#stimuli\silct\silct_npx_1416\0932.jpg',0)
    # input_image = cv2.bitwise_not(img)
    output_image = "smoothed_output.png"
    
    # 处理参数说明：
    # smoothness: 平滑强度 (0-1)，值越大越平滑
    # epsilon_factor: 轮廓简化系数 (0.005-0.05)，值越大越简化
    # spline_points: 样条曲线点数 (50-500)，值越大曲线越精细
    
    processed = process_image(
        input_image, 
        output_image,
        smoothness=0.1,        # 中等平滑
        epsilon_factor=0.001,   # 中等简化
        spline_points=500,      # 150个点构成曲线
        show_steps=True         # 显示处理过程
    )
    
    # print(f"处理完成! 结果已保存至 {output_image}")
