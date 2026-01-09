'''

这个脚本用于调整图片对比度，以使得silct具有和boulder相似的对比度。

'''


#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 或者，如果只需要最简单的方法，可以使用以下代码
def simple_contrast_match(img1_path, img2_path, output_path='result.png'):
    """
    简单版本：使用Lab颜色空间将img2的对比度调整到与img1相同
    """
    # 读取图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # 计算img1的对比度（灰度标准差）
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    target_contrast = np.std(img1_gray)
    
    # 使用Lab颜色空间调整img2
    lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 调整L通道
    mean_l = np.mean(l)
    std_l = np.std(l)
    if std_l > 1e-10:
        l_adjusted = mean_l + (target_contrast / std_l) * (l - mean_l)
        l_adjusted = np.clip(l_adjusted, 0, 255).astype(np.uint8)
    else:
        l_adjusted = l
    
    # 合并通道并转换回BGR
    lab_adjusted = cv2.merge([l_adjusted, a, b])
    result = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)
    
    # 保存结果
    cv2.imwrite(output_path, result)
    
    # 打印对比度信息
    original_contrast = np.std(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    new_contrast = np.std(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
    
    print(f"目标对比度 (img1): {target_contrast:.2f}")
    print(f"原始对比度 (img2): {original_contrast:.2f}")
    print(f"调整后对比度: {new_contrast:.2f}")
    
    return result

# 使用简单版本
# result = simple_contrast_match('img1.png', 'img2.png', 'output.png')



#%%

result = simple_contrast_match('GoodOne!.png', 'GoodOne_Silct.png', 'output.png')


