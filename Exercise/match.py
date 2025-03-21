import cv2
import numpy as np

# 加载模板图像
template = cv2.imread('mouse.jpg', 0)

# 获取模板图像的尺寸
w, h = template.shape[::-1]

# 打开摄像头
cap = cv2.VideoCapture(2)

# 设置匹配阈值
threshold = 0.7

def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    
    # 转换为float类型
    boxes = np.array(boxes).astype("float")
    
    # 初始化选中的索引列表
    pick = []
    
    # 获取坐标
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    # 计算面积
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # 根据置信度排序
    idxs = np.argsort(boxes[:,4])
    
    while len(idxs) > 0:
        # 将最后一个框加入选中列表
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # 找出最大重叠区域
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # 计算重叠区域的宽度和高度
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # 计算重叠率
        overlap = (w * h) / area[idxs[:last]]
        
        # 删除重叠区域大于阈值的框
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    
    return boxes[pick].astype("int")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    
    # 找到匹配位置
    loc = np.where(res >= threshold)
    
    # 构建检测框列表
    boxes = []
    for pt in zip(*loc[::-1]):
        boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h, res[pt[1], pt[0]]])
    
    # 应用非极大值抑制
    if boxes:
        boxes = non_max_suppression(np.array(boxes), 0.3)
        
        # 绘制筛选后的框
        for box in boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    
    cv2.imshow('Matching Result', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()