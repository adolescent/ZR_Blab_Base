'''
手动调整后可以得到封闭边界。对这个边界再次进行边界识别和剪影效果会比较好。

'''

#%%

import cv2
import numpy as np
import OS_Tools as ot
import matplotlib.pyplot as plt
from tqdm import tqdm


# 1. 读取图片 (以灰度模式读取)
# 请将 'input.png' 替换为你的文件名
# img = cv2.imread('10143_HD.jpg', cv2.IMREAD_GRAYSCALE)
# --- 配置区域 ---
def Boulder_Silct_New(img_path,boulder_path,silct_path,thres=120,min_area=100):
    image_path = img_path     # 你的图片路径
    threshold_value = thres         # 阈值：低于此值的像素被认为是线条/前景
    min_area_threshold = min_area      # 【关键新增】最小面积阈值。
                                # 小于这个面积的孤立点将被视为噪点去掉。
                                # 如果噪点没去干净，调大它；如果正常线条被去了，调小它。
    # ----------------

    # 1. 读取图片 (灰度模式)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"错误：无法读取文件 {image_path}")
        exit()

    # 2. 阈值处理，生成二值图
    # 假设线条比背景暗，使用 INV 模式将线条变为白色(255)，背景变为黑色(0)
    ret, binary = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # 3. 查找所有外轮廓
    contours_all, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ============================
    # 【关键新增步骤：过滤噪点】
    # ============================
    valid_contours = []
    for cnt in contours_all:
        # 计算每个轮廓的面积
        area = cv2.contourArea(cnt)
        # 只有面积大于设定的阈值，才算作有效轮廓
        if area > min_area_threshold:
            valid_contours.append(cnt)

    # print(f"原始检测到 {len(contours_all)} 个轮廓，过滤噪点后剩余 {len(valid_contours)} 个有效轮廓。")

    # 4. 准备画布和绘制
    h, w = img.shape
    stroke_canvas = np.full((h, w), 127, dtype=np.uint8) # 描边画布
    fill_canvas   = np.full((h, w), 127, dtype=np.uint8) # 剪影画布

    if valid_contours:
        # 注意：这里我们使用过滤后的 'valid_contours' 进行绘制

        # --- 效果 2: 绘制线宽为4的描边图 ---
        # 颜色0代表黑色，线宽4
        cv2.drawContours(stroke_canvas, valid_contours, -1, 0, 4)

        # --- 效果 3: 填充的剪影图 ---
        # 线宽 -1 代表填充
        cv2.drawContours(fill_canvas, valid_contours, -1, 0, -1)

        # 保存结果
        cv2.imwrite(boulder_path, stroke_canvas)
        cv2.imwrite(silct_path, fill_canvas)
        # print("处理完成！无噪点图像已保存。")
    else:
        print("未检测到符合要求的闭合轮廓，请检查阈值或面积过滤设置。")
    return stroke_canvas,fill_canvas
#%% 运行部分
if __name__ == '__main__':
    all_doodle_path = ot.Get_File_Name(r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\Batch2_HD_Boulder_Raw')
    boulder_savepath = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\_Doodle_Batch2_HD_Boulder'
    silct_savepath = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\_Doodle_Batch2_HD_Silct'
    for i,image_path in tqdm(enumerate(all_doodle_path)):
        # 替换为你的图片路径
        c_filename = image_path.split('\\')[-1]
        counter_filename = ot.Join(boulder_savepath,c_filename)
        silct_filename = ot.Join(silct_savepath,c_filename)
    
        # 提取轮廓和剪影
        contour_img, silhouette_img = Boulder_Silct_New(
            image_path,
            boulder_path=counter_filename,
            silct_path=silct_filename
        )
