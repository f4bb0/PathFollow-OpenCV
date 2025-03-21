import cv2
import numpy as np

# 读取图像
image = cv2.imread('sqare.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 高斯模糊
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# cv2.imshow('blurred', blurred)
# 边缘检测
edges = cv2.Canny(blurred, 50, 150)
cv2.imshow('edges', edges)
# 查找所有轮廓
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到最大的轮廓（外框）
outer_contour = max(contours, key=cv2.contourArea)
epsilon = 0.02 * cv2.arcLength(outer_contour, True)
outer_approx = cv2.approxPolyDP(outer_contour, epsilon, True)

if len(outer_approx) == 4:
    # 绘制外框轮廓
    cv2.drawContours(image, [outer_approx], -1, (0, 255, 0), 2)
    
    # 创建mask，只保留外框内部稍小的区域
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [outer_approx], -1, 255, -1)
    
    # 收缩mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    # 在mask区域内寻找内框
    edges_roi = cv2.bitwise_and(edges, edges, mask=mask)
    cv2.imshow('roi', edges_roi)
    contours_inner, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours_inner:
        # 找到最大的内轮廓
        inner_contour = max(contours_inner, key=cv2.contourArea)
        epsilon_inner = 0.02 * cv2.arcLength(inner_contour, True)
        inner_approx = cv2.approxPolyDP(inner_contour, epsilon_inner, True)
        
        if len(inner_approx) == 4:
            # 绘制内框轮廓
            cv2.drawContours(image, [inner_approx], -1, (255, 0, 0), 2)
            
            # 标记外框角点
            for point in outer_approx.reshape(-1, 2):
                cv2.circle(image, tuple(point), 5, (0, 0, 255), -1)
            
            # 标记内框角点
            for point in inner_approx.reshape(-1, 2):
                cv2.circle(image, tuple(point), 5, (0, 255, 255), -1)

# 显示结果
cv2.imshow('Image with corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()