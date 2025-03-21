import cv2
import numpy as np

# 读取图像
image = cv2.imread('sqare.jpg')
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
epsilon = 0.02 * cv2.arcLength(outer_contour, True)
outer_approx = cv2.approxPolyDP(outer_contour, epsilon, True)

if len(outer_approx) == 4:
    # 绘制外框轮廓
    cv2.drawContours(image, [outer_approx], -1, (0, 255, 0), 2)
    # 获取外框角点坐标
    pts = outer_approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    # 创建mask，只保留外框内部稍小的区域
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [outer_approx], -1, 255, -1)
    
    # 收缩mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    # 在mask区域内寻找内框
    edges_roi = cv2.bitwise_and(edges, edges, mask=mask)
    contours_inner, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours_inner:
        # 找到最大的内轮廓
        inner_contour = max(contours_inner, key=cv2.contourArea)
        epsilon_inner = 0.02 * cv2.arcLength(inner_contour, True)
        inner_approx = cv2.approxPolyDP(inner_contour, epsilon_inner, True)
        
        if len(inner_approx) == 4:
            # 绘制内框轮廓
            cv2.drawContours(image, [inner_approx], -1, (255, 0, 0), 2)
            # 获取内框角点坐标
            pts = inner_approx.reshape(4, 2)
            rect2 = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect2[0] = pts[np.argmin(s)]  # 左上
            rect2[2] = pts[np.argmax(s)]  # 右下
            diff = np.diff(pts, axis=1)
            rect2[1] = pts[np.argmin(diff)]  # 右上
            rect2[3] = pts[np.argmax(diff)] 
            #标记外框角点
            for point in outer_approx.reshape(-1, 2):
                cv2.circle(image, tuple(point), 5, (0, 0, 255), -1)
            
            #标记内框角点
            for point in inner_approx.reshape(-1, 2):
                cv2.circle(image, tuple(point), 5, (0, 255, 255), -1)
            rect3 = np.zeros((4, 2), dtype="float32")
            for i, point in enumerate(rect):
                # Calculate exact midpoint between outer and inner corners
                rect3[i] = np.array([
                    (rect[i][0] + rect2[i][0]) / 2,
                    (rect[i][1] + rect2[i][1]) / 2
                ])
                # print(f"外角点 Outer{i+1} 的坐标: {tuple(map(int, point))}")
            
            # for i, point in enumerate(rect2):
            #     print(f"内角点 Inner{i+1} 的坐标: {tuple(map(int, point))}")
                
            for i, point in enumerate(rect3):
                # Convert the floating point coordinates to integers for drawing
                center_point = tuple(map(int, rect3[i]))
                cv2.circle(image, center_point, 5, (0, 0, 255), -1)
                print(f"中心点 Center{i+1} 的坐标: {center_point}")
# 显示结果
cv2.imshow('Image with corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()