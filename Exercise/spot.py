import cv2
import numpy as np

def nothing(x):
    pass

# 创建一个窗口
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 800, 600)

# 创建HSV阈值的滑动条
cv2.createTrackbar('H_min', 'image', 0, 180, nothing)
cv2.createTrackbar('S_min', 'image', 0, 255, nothing)
cv2.createTrackbar('V_min', 'image', 0, 255, nothing)
cv2.createTrackbar('H_max', 'image', 10, 180, nothing)
cv2.createTrackbar('S_max', 'image', 255, 255, nothing)
cv2.createTrackbar('V_max', 'image', 255, 255, nothing)
#上面是滑动框调试用的

# 初始化摄像头
cap = cv2.VideoCapture(2)

while True:
    # 读取摄像头画面
    ret, frame = cap.read()
    if not ret:
        break
        
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 获取滑动条的值
    H_min = cv2.getTrackbarPos('H_min', 'image')
    S_min = cv2.getTrackbarPos('S_min', 'image')
    V_min = cv2.getTrackbarPos('V_min', 'image')
    H_max = cv2.getTrackbarPos('H_max', 'image')
    S_max = cv2.getTrackbarPos('S_max', 'image')
    V_max = cv2.getTrackbarPos('V_max', 'image')

    # 定义激光点的HSV颜色范围
    lower_red = np.array([H_min, S_min, V_min])
    upper_red = np.array([H_max, S_max, V_max])

    # 创建掩码
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # 进行形态学操作以去除噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 复制原始帧以便绘制
    frame_with_points = frame.copy()

    if contours:
        # 过滤轮廓以去除噪声
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 2]
        
        if valid_contours:
            # 找到面积最大的轮廓
            largest_contour = max(valid_contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                new_center = (cx, cy)#坐标！！！！！！！！cx和cy
                cv2.circle(frame_with_points, new_center, 5, (0, 255, 0), -1)

    # 显示图片
    cv2.imshow('frame', frame_with_points)
    cv2.imshow('mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()