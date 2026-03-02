'''
把图片的背景去掉，换成127的灰色背景。

'''

#%%
from tqdm import tqdm
import OS_Tools as ot
import matplotlib.pyplot as plt
import numpy as np
import cv2

def process_image(image_path, output_path):
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片，请检查路径")
        return

    # 2. 转换为灰度图，用于创建掩码
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 阈值分割 (假设背景接近纯白，240-255之间的认为是背景)
    # 如果背景有色差，可以调整 240 这个阈值
    _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)

    # 4. 边缘处理：腐蚀 (Erosion)
    # 腐蚀操作会使白色区域（物体）缩小，从而去除物体边缘残留的白边
    # 3-5像素的腐蚀，我们使用 5x5 或 7x7 的卷积核
    kernel_size = 5 # 调整此数值可以控制腐蚀强度
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # 5. 边缘处理：羽化 (Gaussian Blur)
    # 将遮罩转换为浮点数进行模糊处理，产生平滑的过度
    mask_float = eroded_mask.astype(float) / 255.0
    # ksize 必须是奇数，数值越大，边缘越柔和
    feathered_mask = cv2.GaussianBlur(mask_float, (7, 7), 0)

    # 6. 准备背景和合成
    # 创建纯 127 灰色的背景图
    bg_color = 127
    background = np.full(img.shape, bg_color, dtype=np.uint8)

    # 将 mask 扩展为 3 通道 (H, W, 3) 方便计算
    alpha = cv2.merge([feathered_mask, feathered_mask, feathered_mask])

    # 7. 线性插值合成: Output = Image * alpha + Background * (1 - alpha)
    # 这确保了边缘是物体色和背景色的平滑过渡
    foreground = img.astype(float)
    background = background.astype(float)
    
    final_img = cv2.multiply(alpha, foreground) + cv2.multiply(1.0 - alpha, background)
    final_img = final_img.astype(np.uint8)

    # 8. 保存结果
    cv2.imwrite(output_path, final_img)
    # print(f"处理完成，结果已保存至: {output_path}")



def process_outer_boundary(image_path, output_path,threshold= 170):
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None: return

    # 2. 预处理：转灰度并做简单的阈值处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 稍微模糊一下有助于消除噪点对轮廓的影响
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 获取初步二值图，重点是区分背景和物体
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

    # 3. 提取轮廓
    # RETR_EXTERNAL 保证只寻找最外层轮廓，忽略物体内部的孔洞
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("未检测到有效轮廓")
        return

    # 4. 创建实心的外轮廓掩码
    # 这一步能确保物体内部即便有白色，也会被填充为“物体”区域
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # 5. 腐蚀处理 (向内收缩 3-5 像素)
    # 消除残留白边
    kernel = np.ones((7,7), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # 6. 高斯羽化 (让边缘过渡自然)
    mask_float = eroded_mask.astype(float) / 255.0
    feathered_mask = cv2.GaussianBlur(mask_float, (7,7), 1)
    alpha = cv2.merge([feathered_mask] * 3)

    # 7. 合成背景
    bg_color = 127
    background = np.full(img.shape, bg_color, dtype=np.uint8).astype(float)
    foreground = img.astype(float)

    # 合成公式
    final_img = foreground * alpha + background * (1.0 - alpha)
    final_img = final_img.astype(np.uint8)

    # 8. 保存
    cv2.imwrite(output_path, final_img)
    # print(f"处理完成！")

#%%%%  方法3，简单背景的边界识别。

def remove_background_with_feathering(image_path, output_path, gray_value=127, feather_radius=2, dilation_size=2):
    """
    识别物体边界，移除背景并羽化边缘
    
    参数:
        image_path: 输入图像路径
        output_path: 输出图像路径
        gray_value: 背景填充的灰度值 (默认127)
        feather_radius: 羽化半径 (默认2像素)
        dilation_size: 向内填充像素数 (默认2像素)
    """
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图像")
        return
    
    # 2. 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. 使用自适应阈值处理渐变背景
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 4. 形态学操作去除噪声
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 5. 找到物体轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("未找到物体轮廓")
        return
    
    # 6. 创建掩码（物体区域为白色）
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, -1)
    
    # 7. 向内腐蚀掩码（避免边缘残留背景）
    if dilation_size > 0:
        kernel = np.ones((2*dilation_size+1, 2*dilation_size+1), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    
    # 8. 创建羽化边缘
    if feather_radius > 0:
        # 高斯模糊创建羽化效果
        blurred_mask = cv2.GaussianBlur(mask, (2*feather_radius+1, 2*feather_radius+1), 0)
        # 转换为0-1范围的浮点掩码
        mask_float = blurred_mask.astype(np.float32) / 255.0
        
        # 9. 应用羽化掩码
        result = img.astype(np.float32)
        for c in range(3):
            result[:, :, c] = result[:, :, c] * mask_float + gray_value * (1 - mask_float)
        result = result.astype(np.uint8)
    else:
        # 不使用羽化
        result = img.copy()
        result[mask == 0] = gray_value
    
    # 10. 保存结果
    cv2.imwrite(output_path, result)
    # print(f"处理完成，结果保存到: {output_path}")
    
    # # 可选：显示结果
    # cv2.imshow('Original', img)
    # cv2.imshow('Mask', mask)
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


#%% 使用示例
if __name__ == "__main__":

    savepath = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\AI_Regeneration_1_GrayBK'
    rawpath = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\AI_Regeneration_1'
    all_graph_name = ot.Get_File_Name(rawpath)
    for i,cimg in tqdm(enumerate(all_graph_name)):
        c_imgname = cimg.split('\\')[-1]
        output1 = ot.Join(savepath,c_imgname)
        remove_background_with_feathering(cimg,output1)

    
    # # 方法1: 基础方法（简单快速）
    # result1 = remove_background_and_replace(input_image, "output_basic.jpg")
    # # 方法2: 使用Alpha混合（效果更自然）
    # result2 = process_with_alpha_blending(input_image, "output_alpha.jpg")
    # # 方法3: 使用轮廓检测（适合有明显物体的图片）
    # result3 = remove_background_with_refined_edge(input_image, "output_refined.jpg")




    
