import cv2
import numpy as np
image = cv2.imread('bird.jpg')
image[50:150, 50:150] = [0, 255, 0]


# 检查图像是否成功读取
if image is None:
    print("错误：无法加载图像，请检查路径是否正确。")
    exit()

#----------------------------------USER DEFINED CODE-------------------------------
kernel_size=55
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.blur(image, (kernel_size, kernel_size))
edges = cv2.Canny(image, 100, 200)
# Define the kernel
kernel = np.ones((5, 5), np.uint8)
image = cv2.dilate(image, kernel, iterations=1)


# 2. 显示图像
# 创建一个名为 "Display Image" 的窗口，并在其中显示图像
cv2.imshow("My Image", edges)
# 3. 等待用户按键
# 参数 0 表示无限等待，直到用户按下任意键
key = cv2.waitKey(10000)

# 4. 根据用户按键执行操作
if key == ord('s'):  # 如果按下 's' 键
    # 保存图像
    output_path = "saved_image.png"
    cv2.imwrite(output_path, image)
    print(f"图像已保存为 {output_path}")
else:  # 如果按下其他键
    print("图像未保存。")

# 5. 关闭所有窗口
cv2.destroyAllWindows()