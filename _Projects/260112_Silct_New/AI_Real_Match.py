'''
操作之前，先对ai图进行质量检查。包括明显有问题的，需要再生成一下。

用这个脚本把图片匹配，需要提供：
1-简笔画的silct
2-AI图的silct，用一样的方式生成，需要一些微调
3-AI图，背景最好是纯白色的，方便后续调整。

'''

#%%
from tqdm import tqdm
import OS_Tools as ot
import cv2
import numpy as np

def calculate_transform_matrix(mask1, mask2, max_shift=50, max_scale=1.5):
    """
    计算mask2到mask1的平移和缩放变换矩阵
    
    参数:
        mask1: 参考mask图像
        mask2: 需要变换的mask图像
        max_shift: 最大平移像素数
        max_scale: 最大缩放倍数
    
    返回:
        M: 变换矩阵 (2x3)
        shift_x, shift_y: 实际应用的平移量
        scale: 实际应用的缩放倍数
    """
    # 二值化处理：将物体(0)设为255，背景(127)设为0
    _, mask1_bin = cv2.threshold(mask1, 50, 255, cv2.THRESH_BINARY_INV)
    _, mask2_bin = cv2.threshold(mask2, 50, 255, cv2.THRESH_BINARY_INV)
    
    # 找到物体的轮廓
    contours1, _ = cv2.findContours(mask1_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours1 or not contours2:
        print("未检测到物体轮廓")
        return None, 0, 0, 1.0
    
    # 找到最大的轮廓
    contour1 = max(contours1, key=cv2.contourArea)
    contour2 = max(contours2, key=cv2.contourArea)
    
    # 计算最小外接矩形
    rect1 = cv2.minAreaRect(contour1)
    rect2 = cv2.minAreaRect(contour2)
    
    # 获取矩形的中心点
    center1 = np.array(rect1[0])
    center2 = np.array(rect2[0])
    
    # 获取矩形的宽度和高度
    width1, height1 = rect1[1]
    width2, height2 = rect2[1]
    
    # 计算缩放比例
    scale_x = width1 / width2
    scale_y = height1 / height2
    scale = (scale_x + scale_y) / 2  # 使用平均缩放比例
    
    # 限制缩放倍数
    scale = np.clip(scale, 1/max_scale, max_scale)
    
    # 计算平移量
    shift_x = center1[0] - center2[0]
    shift_y = center1[1] - center2[1]
    
    # 限制平移量
    shift_x = np.clip(shift_x, -max_shift, max_shift)
    shift_y = np.clip(shift_y, -max_shift, max_shift)
    
    # 构建变换矩阵：先缩放，后平移
    # 缩放矩阵
    scale_matrix = np.array([[scale, 0, 0],
                             [0, scale, 0]])
    
    # 平移矩阵
    shift_matrix = np.array([[1, 0, shift_x],
                             [0, 1, shift_y]])
    
    # 组合变换矩阵（先缩放后平移）
    M = np.dot(shift_matrix, scale_matrix)
    
    print(f"应用变换: 平移({shift_x:.2f}, {shift_y:.2f}), 缩放{scale:.4f}")
    return M, shift_x, shift_y, scale

def apply_transform_to_image(img, M, output_size=None):
    """
    应用变换矩阵到图像
    
    参数:
        img: 输入图像
        M: 变换矩阵
        output_size: 输出图像大小，默认与输入图像相同
    
    返回:
        transformed_img: 变换后的图像
    """
    if output_size is None:
        output_size = (img.shape[1], img.shape[0])
    
    # 应用仿射变换
    transformed_img = cv2.warpAffine(img, M, output_size, 
                                     flags=cv2.INTER_LINEAR, 
                                     borderMode=cv2.BORDER_REFLECT)
    
    return transformed_img

#%% ### 以下是运行部分，根据实际需求调整。
def main():
    # 读取图像
    mask1 = cv2.imread('mask1.png', cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread('mask2.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('img2.png')
    
    if mask1 is None or mask2 is None or img2 is None:
        print("请确保所有图像文件都存在")
        return
    
    # 设置参数（可调整）
    max_shift = 50      # 最大平移像素数
    max_scale = 1.5     # 最大缩放倍数
    
    # 计算变换矩阵
    M, shift_x, shift_y, scale = calculate_transform_matrix(
        mask1, mask2, max_shift, max_scale
    )
    
    if M is None:
        return
    
    # 应用变换到mask2和img2
    transformed_mask2 = apply_transform_to_image(mask2, M)
    transformed_img2 = apply_transform_to_image(img2, M)
    
    # 显示结果
    cv2.imshow('Original Mask1', mask1)
    cv2.imshow('Original Mask2', mask2)
    cv2.imshow('Transformed Mask2', transformed_mask2)
    cv2.imshow('Original Image2', img2)
    cv2.imshow('Transformed Image2', transformed_img2)
    
    # 保存结果
    cv2.imwrite('transformed_mask2.png', transformed_mask2)
    cv2.imwrite('transformed_img2.png', transformed_img2)
    
    print("处理完成！结果已保存")
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_alignment(mask1, mask2, transformed_mask2):
    """
    可视化对齐效果（可选功能）
    """
    # 创建彩色可视化图像
    height, width = mask1.shape
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 二值化mask
    _, mask1_bin = cv2.threshold(mask1, 50, 255, cv2.THRESH_BINARY_INV)
    _, mask2_bin = cv2.threshold(transformed_mask2, 50, 255, cv2.THRESH_BINARY_INV)
    
    # 红色表示mask1，绿色表示mask2，黄色表示重叠区域
    vis[:,:,2] = mask1_bin  # 红色通道
    vis[:,:,1] = mask2_bin  # 绿色通道
    
    # 黄色重叠区域
    overlap = cv2.bitwise_and(mask1_bin, mask2_bin)
    vis[:,:,0] = overlap  # 蓝色通道设为0
    vis[:,:,2] = cv2.add(vis[:,:,2], overlap)  # 红色+黄色=橙色
    
    cv2.imshow('Alignment Visualization', vis)
    cv2.imwrite('alignment_visualization.png', vis)

if __name__ == "__main__":
    main()