import cv2
import numpy as np

# 读取图像
image = cv2.imread('tsmall2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 高斯模糊
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#cv2.imshow('blurred', blurred)
# 边缘检测
edges = cv2.Canny(blurred, 50, 150)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义结构元素
dilated = cv2.dilate(edges, kernel, iterations=1)  # 膨胀操作
#cv2.imshow('dilated', dilated)

# 查找所有轮廓
contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到最大的轮廓（外框）
outer_contour = max(contours, key=cv2.contourArea)
# 拟合外部椭圆
if len(outer_contour) >= 5:  # 至少需要5个点才能拟合椭圆
    outer_ellipse = cv2.fitEllipse(outer_contour)
    # 绘制外部椭圆
    cv2.ellipse(image, outer_ellipse, (0, 255, 0), 2)
    # 获取外部椭圆参数
    (outer_center, outer_axes, outer_angle) = outer_ellipse
    
    # 创建mask，只保留外框内部稍小的区域
    mask = np.zeros_like(gray)
    cv2.ellipse(mask, outer_ellipse, 255, -1)
    
    # 收缩mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    # 在mask区域内寻找内框
    edges_roi = cv2.bitwise_and(edges, edges, mask=mask)
    contours_inner, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours_inner:
        # 找到最大的内轮廓
        inner_contour = max(contours_inner, key=cv2.contourArea)
        if len(inner_contour) >= 5:  # 至少需要5个点才能拟合椭圆
            inner_ellipse = cv2.fitEllipse(inner_contour)
            # 绘制内部椭圆
            cv2.ellipse(image, inner_ellipse, (255, 0, 0), 2)
            # 获取内部椭圆参数
            (inner_center, inner_axes, inner_angle) = inner_ellipse
            
            # 计算中间椭圆的参数
            middle_center = ((outer_center[0] + inner_center[0])/2, 
                           (outer_center[1] + inner_center[1])/2)
            middle_axes = ((outer_axes[0] + inner_axes[0])/2,
                         (outer_axes[1] + inner_axes[1])/2)
            middle_angle = (outer_angle + inner_angle)/2
            
            # 绘制中间椭圆
            middle_ellipse = (middle_center, middle_axes, middle_angle)
            cv2.ellipse(image, middle_ellipse, (0, 255, 255), 2)
            
            # 标记椭圆中心点
            cv2.circle(image, (int(outer_center[0]), int(outer_center[1])), 5, (0, 0, 255), -1)
            cv2.circle(image, (int(middle_center[0]), int(middle_center[1])), 5, (255, 0, 255), -1)
            cv2.circle(image, (int(inner_center[0]), int(inner_center[1])), 5, (0, 255, 255), -1)
            
            # 打印椭圆信息
            print("外部椭圆:")
            print(f"中心点: {outer_center}")
            print(f"长轴和短轴: {outer_axes}")
            print(f"旋转角度: {outer_angle}")
            print("\n中间椭圆:")
            print(f"中心点: {middle_center}")
            print(f"长轴和短轴: {middle_axes}")
            print(f"旋转角度: {middle_angle}")
            print("\n内部椭圆:")
            print(f"中心点: {inner_center}")
            print(f"长轴和短轴: {inner_axes}")
            print(f"旋转角度: {inner_angle}")

# 显示结果
cv2.imshow('Image with corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()