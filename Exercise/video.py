import cv2
import time

# 创建 VideoCapture 对象，读取摄像头视频
cap = cv2.VideoCapture(2)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# 读取视频帧
while True:
    ret, frame = cap.read()
    
    # 如果读取到最后一帧，退出循环
    if not ret:
        break
    
    # 显示当前帧
    cv2.imshow('Camera', frame)
    
    # 获取按键输入
    key = cv2.waitKey(25) & 0xFF
    
    # 按下 'q' 键退出
    if key == ord('q'):
        break
    
    # 按下 's' 键保存当前帧
    if key == ord('s'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'frame_{timestamp}.jpg'
        cv2.imwrite(filename, frame)
        print(f"Frame saved as {filename}")

# 释放资源
cap.release()
cv2.destroyAllWindows()