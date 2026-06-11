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
import matplotlib.pyplot as plt

def align_images_ecc(mask1, mask2, img,force_trans = True):
    """
    使用 ECC 算法将 mask2 对齐到 mask1，并变换 img
    """
    # 1. 预处理：将 mask 转换为 float32 格式，这是 ECC 算法的要求
    # 如果 mask 是 0-255，转换为 0-1 更有利于计算
    sz = mask1.shape
    m1 = mask1.astype(np.float32)
    m2 = mask2.astype(np.float32)

    # 2. 平滑处理（关键步）：
    # 模糊可以消除“长边界”噪声的干扰，让算法关注大体轮廓的重合
    m1_blur = cv2.GaussianBlur(m1, (5, 5), 0)
    m2_blur = cv2.GaussianBlur(m2, (5, 5), 0)

    # 3. 初始化变换矩阵
    # 使用 MOTION_AFFINE 模型。虽然它包含旋转，但在仅有平移/缩放差异时表现最稳
    # 矩阵形状为 2x3
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # 4. 定义迭代终止条件
    # 迭代 500 次或参数变化小于 1e-7
    number_of_iterations = 500
    termination_eps = 1e-7
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                number_of_iterations, termination_eps)

    try:
        # 5. 运行 ECC 算法
        # 结果会更新 warp_matrix
        (cc, warp_matrix) = cv2.findTransformECC(m1_blur, m2_blur, warp_matrix, 
                                                 cv2.MOTION_AFFINE, criteria, 
                                                 None, 5) # 最后一个参数是高斯金字塔层数，5有助于处理大位移
    except cv2.error as e:
        print(f"ECC 对齐失败: {e}. 请检查 mask 是否有重叠区域。")
        # 如果失败，返回单位阵
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # 6. 将矩阵应用到原始图片 img
    # 目标尺寸为 mask1 的尺寸
    rows, cols = mask1.shape[:2]
    
    # 变换图片，空白区域填充 255 (白色)
    if force_trans:
        warp_matrix = refine_matrix_to_scale_translate(warp_matrix)
    aligned_img = cv2.warpAffine(img, warp_matrix, (cols, rows), 
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, 
                                 borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=(255, 255, 255))
    
    # 注意：ECC 计算的是从 mask1 到 mask2 的映射，
    # 所以在 warpAffine 中通常需要配合 WARP_INVERSE_MAP 使用，或者根据需要调整。
    # 这里直接返回得到的变换矩阵
    return warp_matrix, aligned_img

# --- 进阶技巧：强制限制为仅平移和缩放 ---
def refine_matrix_to_scale_translate(M):
    """
    如果 M 包含微小旋转，可手动将非主对角线元素置 0 以强制锁定为平移和缩放
    """
    refined_M = M.copy()
    refined_M[0, 1] = 0 # 移除旋转/剪切影响
    refined_M[1, 0] = 0
    return refined_M

# --- 应用变换的代码，可以手动自己做----
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
if __name__ == "__main__":
    # 读取图像
    all_mats = np.zeros(shape=(400,2,3))
    savepath = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\_AI_Real_3_Size_adj'
    target_path =r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\HD_Silct'
    aim_path = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\AI_Real_3_Silct'
    img_path = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\AI_Real_3'
    all_mask1 = ot.Get_File_Name(target_path)
    all_mask2 = ot.Get_File_Name(aim_path)
    all_img = ot.Get_File_Name(img_path)
    for i in tqdm(range(400)):
        mask1 = cv2.imread(all_mask1[i], cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(all_mask2[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(all_img[i])
        filename = all_img[i].split('\\')[-1]
        warp_matrix, aligned_img = align_images_ecc(mask1,mask2,img2,force_trans=False)
        all_mats[i,:,:] = warp_matrix # 保存变换矩阵
        
        real_savepath = ot.Join(savepath,filename)
        cv2.imwrite(real_savepath,aligned_img)
