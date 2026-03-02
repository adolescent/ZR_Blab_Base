'''
Ignore other method, cut boulders into dashed line.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_to_dashed_contour(contour, dash_length=10, gap_length=5, min_segment_length=3):
    """
    将实线轮廓转换为等长虚线
    
    参数:
    contour: 输入轮廓 (N, 1, 2)
    dash_length: 虚线段的长度(像素)
    gap_length: 间隔的长度(像素)
    min_segment_length: 最小线段长度(像素)，用于处理轮廓尾部
    
    返回:
    dashed_contour: 虚线轮廓点集 (M, 1, 2)
    """
    # 将轮廓点展平为(N, 2)格式
    points = contour.reshape(-1, 2)
    
    # 计算轮廓总长度和每段长度
    segment_lengths = []
    total_length = 0
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i+1) % len(points)]
        seg_len = np.linalg.norm(p2 - p1)
        segment_lengths.append(seg_len)
        total_length += seg_len
    
    # 如果轮廓太小，直接返回原轮廓
    if total_length < dash_length + gap_length:
        return contour
    
    # 创建虚线点集
    dashed_points = []
    current_pos = 0.0  # 当前位置在轮廓上的长度
    is_dash = True     # 当前状态：虚线或间隔
    
    # 遍历轮廓生成虚线
    while current_pos < total_length:
        if is_dash:
            # 虚线模式：生成一个dash_length长的虚线
            segment_end = min(current_pos + dash_length, total_length)
            
            # 处理尾部过短的情况
            if segment_end - current_pos < min_segment_length:
                break
                
            # 生成虚线段的点
            dash_points = []
            start_point = get_point_on_contour(points, segment_lengths, current_pos)
            end_point = get_point_on_contour(points, segment_lengths, segment_end)
            
            # 添加虚线起点
            dash_points.append(start_point)
            
            # 如果虚线跨越多段，添加中间点
            current_segment = current_pos
            while current_segment < segment_end:
                # 找到当前所在的线段
                seg_index, seg_pos = find_segment(segment_lengths, current_segment)
                
                # 计算线段结束点
                segment_end_point = min(segment_end, current_segment + segment_lengths[seg_index] - seg_pos)
                
                # 添加中间点
                if segment_end_point < segment_end:
                    point = get_point_on_contour(points, segment_lengths, segment_end_point)
                    dash_points.append(point)
                
                current_segment = segment_end_point
            
            # 添加虚线终点
            dash_points.append(end_point)
            
            # 添加到虚线轮廓
            for p in dash_points:
                dashed_points.append(p)
            
            current_pos = segment_end
            is_dash = False
        else:
            # 间隔模式：跳过gap_length
            current_pos = min(current_pos + gap_length, total_length)
            is_dash = True
    
    # 转换为OpenCV轮廓格式
    if len(dashed_points) == 0:
        return contour
    
    dashed_contour = np.array(dashed_points).reshape(-1, 1, 2).astype(np.int32)
    return dashed_contour

def find_segment(segment_lengths, position):
    """
    找到给定位置所在的线段
    
    参数:
    segment_lengths: 各线段长度列表
    position: 在轮廓上的位置
    
    返回:
    segment_index: 线段索引
    segment_position: 在线段上的位置
    """
    cumulative = 0
    for i, seg_len in enumerate(segment_lengths):
        if position < cumulative + seg_len:
            return i, position - cumulative
        cumulative += seg_len
    return len(segment_lengths) - 1, segment_lengths[-1]

def get_point_on_contour(points, segment_lengths, position):
    """
    获取轮廓上指定位置的点
    
    参数:
    points: 轮廓点集
    segment_lengths: 各线段长度列表
    position: 在轮廓上的位置
    
    返回:
    point: 位置对应的点坐标
    """
    seg_index, seg_pos = find_segment(segment_lengths, position)
    
    # 获取当前线段的起点和终点
    p1 = points[seg_index]
    p2 = points[(seg_index + 1) % len(points)]
    
    # 计算线段上的位置比例
    seg_len = segment_lengths[seg_index]
    if seg_len > 0:
        t = seg_pos / seg_len
    else:
        t = 0
    
    # 线性插值得到点坐标
    x = p1[0] + t * (p2[0] - p1[0])
    y = p1[1] + t * (p2[1] - p1[1])
    
    return np.array([x, y])

def adaptive_dash_length(total_length, min_dash=5, max_dash=30, num_segments=20):
    """
    根据轮廓长度自适应计算虚线长度
    
    参数:
    total_length: 轮廓总长度
    min_dash: 最小虚线长度
    max_dash: 最大虚线长度
    num_segments: 期望的虚线数量
    
    返回:
    dash_length: 自适应虚线长度
    """
    dash_length = total_length / num_segments
    return np.clip(dash_length, min_dash, max_dash)

def process_image_to_dashed(input_path, output_path, 
                           dash_length=10, 
                           gap_length=5,
                           min_segment_length=3,
                           adaptive=True,
                           show_steps=True):
    """
    处理图像：将实线轮廓转换为虚线
    
    参数:
    input_path: 输入图像路径
    output_path: 输出图像路径
    dash_length: 虚线段的长度(像素)
    gap_length: 间隔的长度(像素)
    min_segment_length: 最小线段长度(像素)
    adaptive: 是否自适应虚线长度
    show_steps: 是否显示处理过程
    """
    # 读取图像
    img = cv2.imread(input_path)
    # img = cv2.bitwise_not(img)
    if img is None:
        raise ValueError(f"无法读取图像: {input_path}")
    
    # 转换为灰度图并二值化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 提取轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建结果图像
    result = np.ones_like(img) * 255
    
    # 处理每个轮廓
    all_dashed_contours = []
    for contour in contours:
        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)
        
        # 自适应虚线长度
        if adaptive:
            contour_dash_length = adaptive_dash_length(perimeter)
        else:
            contour_dash_length = dash_length
        
        # 转换为虚线
        dashed_contour = convert_to_dashed_contour(
            contour, 
            dash_length=contour_dash_length,
            gap_length=gap_length,
            min_segment_length=min_segment_length
        )
        
        all_dashed_contours.append(dashed_contour)
        
        # 绘制虚线轮廓
        cv2.drawContours(result, [dashed_contour], -1, (0, 0, 0), 2)
    
    # 保存结果
    # cv2.imwrite(output_path, result)
    
    # 显示处理过程
    if show_steps:
        plt.figure(figsize=(15, 8))
        
        # 原始图像
        plt.subplot(231)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Raw")
        
        # 二值化图像
        plt.subplot(232)
        plt.imshow(binary, cmap='gray')
        plt.title("Bin")
        
        # 轮廓提取
        contour_img = np.ones_like(img) * 255
        cv2.drawContours(contour_img, contours, -1, (0, 0, 0), 2)
        plt.subplot(233)
        plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        plt.title("Contour")
        
        # 虚线结果
        plt.subplot(234)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("Dashed ")
        
        # 虚线细节放大
        plt.subplot(235)
        if len(all_dashed_contours) > 0:
            # 获取第一个轮廓的边界框
            x, y, w, h = cv2.boundingRect(all_dashed_contours[0])
            if w > 0 and h > 0:
                # 扩大区域
                exp = 20
                x1 = max(0, x - exp)
                y1 = max(0, y - exp)
                x2 = min(img.shape[1], x + w + exp)
                y2 = min(img.shape[0], y + h + exp)
                
                detail_img = result[y1:y2, x1:x2]
                plt.imshow(cv2.cvtColor(detail_img, cv2.COLOR_BGR2RGB))
                plt.title("Dashed Detail")
        
        plt.tight_layout()
        plt.show()
    
    return result

#%% 使用示例
if __name__ == "__main__":
    input_image = r"D:\#Data\#stimuli\silct\silct_npx_1416\0998.jpg"  # 使用之前平滑后的图像
    output_image = "dashed_sketch.png"
    
    # 处理参数说明：
    # dash_length: 虚线段的长度(像素) - 如果adaptive=True，此参数将被覆盖
    # gap_length: 间隔的长度(像素)
    # min_segment_length: 最小线段长度(像素) - 用于处理轮廓尾部
    # adaptive: 是否根据轮廓长度自适应虚线长度
    
    dashed_image = process_image_to_dashed(
        input_image, 
        output_image,
        dash_length=10,          # 虚线长度
        gap_length=20,             # 间隔长度
        min_segment_length=30,     # 最小线段长度
        adaptive=True,            # 自适应虚线长度
        show_steps=True
    )
    
    print(f"虚线转换完成! 结果已保存至 {output_image}")
# %%
