import cv2
import numpy as np
import threading
import time

def nothing(x):
    pass

class FrameProcessor:
    def __init__(self):
        self.frame = None
        self.frame_lock = threading.RLock()
        self.corners = []
        self.running = True
        self.display_frame = None
        self.perspective_matrix = None
        self.warped_size = (400, 400)  # Output size after perspective transform
        self.is_setting_perspective = False
        self.perspective_points = []
        self.bright_center = None  # Store the bright spot center coordinates
        self.last_display = None  # Store last valid display frame
        self.paused = False  # Add pause flag
        self.corners_outer = None
        self.corners_inner = None
        self.midpoints = None
        self.draw_lock = threading.RLock()
        self.coordinates_lock = threading.RLock()
        self.last_valid_midpoints = None  # Store last valid midpoints
        self.last_bright_center = None  # Add this to track changes in bright spot position
        self.bright_center_updated = False  # Replace Event with flag
        self.bright_center_lock = threading.RLock()  # Add lock for flag access
        self.center_point = None  # Intersection of diagonals
        self.target_point = None  # Intersection of rotating line with quadrilateral
        self.angle = 0  # Current angle of rotating line
        self.angular_velocity = 10  # Degrees per second
        self.hsv_window_created = False  # Add flag for window creation

    def update_frame(self, new_frame):
        # Minimize critical section
        warped = None
        with self.frame_lock:
            if self.perspective_matrix is not None and not self.is_setting_perspective:
                warped = cv2.warpPerspective(new_frame, self.perspective_matrix, self.warped_size)
            
        if warped is not None:
            with self.frame_lock:
                self.frame = warped.copy()
        else:
            with self.frame_lock:
                self.frame = new_frame.copy()

    def get_frame(self):
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None

    def process_corners(self):
        while self.running:
            if self.paused:  # Skip processing when paused
                time.sleep(0.1)
                continue
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(edges, kernel, iterations=1)

            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                time.sleep(1)
                continue

            # Process outer contour
            outer_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(outer_contour, True)
            outer_approx = cv2.approxPolyDP(outer_contour, epsilon, True)

            if len(outer_approx) == 4:
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [outer_approx], -1, 255, -1)
                kernel = np.ones((5,5), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
                
                edges_roi = cv2.bitwise_and(edges, edges, mask=mask)
                contours_inner, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours_inner:
                    inner_contour = max(contours_inner, key=cv2.contourArea)
                    epsilon_inner = 0.02 * cv2.arcLength(inner_contour, True)
                    inner_approx = cv2.approxPolyDP(inner_contour, epsilon_inner, True)
                    
                    if len(inner_approx) == 4:
                        # Process corners
                        pts_outer = outer_approx.reshape(4, 2)
                        pts_inner = inner_approx.reshape(4, 2)
                        rect = np.zeros((4, 2), dtype="float32")
                        rect2 = np.zeros((4, 2), dtype="float32")
                        
                        # Sort corners
                        s = pts_outer.sum(axis=1)
                        rect[0] = pts_outer[np.argmin(s)]
                        rect[2] = pts_outer[np.argmax(s)]
                        diff = np.diff(pts_outer, axis=1)
                        rect[1] = pts_outer[np.argmin(diff)]
                        rect[3] = pts_outer[np.argmax(diff)]
                        
                        s = pts_inner.sum(axis=1)
                        rect2[0] = pts_inner[np.argmin(s)]
                        rect2[2] = pts_inner[np.argmax(s)]
                        diff = np.diff(pts_inner, axis=1)
                        rect2[1] = pts_inner[np.argmin(diff)]
                        rect2[3] = pts_inner[np.argmax(diff)]
                        
                        # Calculate midpoints
                        rect3 = np.zeros((4, 2), dtype="float32")
                        for i in range(4):
                            rect3[i] = np.array([
                                (rect[i][0] + rect2[i][0]) / 2,
                                (rect[i][1] + rect2[i][1]) / 2
                            ])
                        
                        # Only update midpoints if detection is valid
                        if all(point.sum() != 0 for point in rect3):  # Check if points are valid
                            with self.draw_lock:
                                self.corners_outer = outer_approx
                                self.corners_inner = inner_approx
                                self.midpoints = rect3  # Current frame midpoints
                                # Only update last_valid_midpoints when detection is successful
                                if rect3 is not None and len(rect3) == 4:
                                    self.last_valid_midpoints = rect3.copy()
                
            time.sleep(1)  # Process once per second

    def process_bright_spot(self):
        """Second thread for bright spot detection"""
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
                
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            # Convert to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Get trackbar values
            h_min = cv2.getTrackbarPos('H_min', 'HSV Controls')
            s_min = cv2.getTrackbarPos('S_min', 'HSV Controls')
            v_min = cv2.getTrackbarPos('V_min', 'HSV Controls')
            h_max = cv2.getTrackbarPos('H_max', 'HSV Controls')
            s_max = cv2.getTrackbarPos('S_max', 'HSV Controls')
            v_max = cv2.getTrackbarPos('V_max', 'HSV Controls')
            
            # Create mask using trackbar values
            lower_red = np.array([h_min, s_min, v_min])
            upper_red = np.array([h_max, s_max, v_max])
            
            # Create mask
            mask = cv2.inRange(hsv, lower_red, upper_red)
            
            # Morphological operations to remove noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Filter contours by area to remove noise
                valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 2]
                
                if valid_contours:
                    # Find the contour with the largest area
                    largest_contour = max(valid_contours, key=cv2.contourArea)
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        new_center = (cx, cy)
                        
                        if new_center != self.bright_center:
                            with self.bright_center_lock:
                                self.bright_center = new_center
                                self.bright_center_updated = True

            time.sleep(0.03)  # ~30fps

    def draw_frame(self):
        """Third thread for all drawing operations"""
        while self.running:
            if self.paused:
                time.sleep(0.03)
                continue
                
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            # Create local copies of shared data
            with self.draw_lock:
                corners_outer = self.corners_outer
                corners_inner = self.corners_inner
                midpoints = self.midpoints
            
            with self.bright_center_lock:
                bright_center = self.bright_center

            # Draw using local copies
            display = frame.copy()
            if corners_outer is not None:
                cv2.drawContours(display, [corners_outer], -1, (0, 255, 0), 2)
            if corners_inner is not None:
                cv2.drawContours(display, [corners_inner], -1, (255, 0, 0), 2)
            if midpoints is not None:
                for point in midpoints:
                    cv2.circle(display, tuple(map(int, point)), 5, (0, 0, 255), -1)
            if bright_center is not None:
                cv2.circle(display, bright_center, 5, (255, 0, 0), -1)
                # Draw center point and target point
                if self.center_point is not None:
                    cv2.circle(display, self.center_point, 5, (255, 255, 0), -1)
                    if self.target_point is not None:
                        cv2.line(display, self.center_point, self.target_point, (0, 255, 255), 2)
                        cv2.circle(display, self.target_point, 5, (0, 255, 255), -1)
            
            with self.frame_lock:
                self.display_frame = display
                self.last_display = display
            
            time.sleep(0.03)  # ~30fps

    def print_coordinates(self):
        """Fourth thread for printing coordinates and calculating target point"""
        def cross_product(v1, v2):
            """计算二维向量叉积"""
            return v1[0] * v2[1] - v1[1] * v2[0]

        def line_intersection(p1, p2, p3, p4):
            """计算两条线段的交点，p1,p2为第一条线段的端点，p3,p4为第二条线段的端点"""
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            
            denominator = ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            if abs(denominator) < 1e-6:  # 平行线
                return None
                
            # 计算交点
            x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
            y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
            
            # 对于对角线交点，不需要检查范围限制
            return (int(x), int(y))

        last_time = time.time()
        self.angular_velocity = 10  # 度/秒
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            midpoints = self.last_valid_midpoints
            if midpoints is not None and not self.paused:
                # 转换midpoints为整数坐标
                midpoints_int = [tuple(map(int, point)) for point in midpoints]
                print("Debug: Processing midpoints:", midpoints_int)
                
                # 计算对角线交点（中心点）
                center = line_intersection(
                    midpoints_int[0], midpoints_int[2],  # 对角线1
                    midpoints_int[1], midpoints_int[3]   # 对角线2
                )
                
                if center:
                    self.center_point = center
                    print(f"Debug: Center found at {center}")
                    
                    # 更新旋转角度
                    self.angle = (self.angle + self.angular_velocity * dt) % 360
                    angle_rad = np.radians(self.angle)
                    
                    # 计算旋转射线的终点（使用足够长的半径）
                    radius = 1000
                    ray_end = (
                        center[0] + int(radius * np.cos(angle_rad)),
                        center[1] + int(radius * np.sin(angle_rad))
                    )

                    def check_intersection(p1, p2, p3, p4):
                        """检查射线与边的交点，带范围检查"""
                        x1, y1 = p1
                        x2, y2 = p2
                        x3, y3 = p3
                        x4, y4 = p4
                        
                        denominator = ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
                        if abs(denominator) < 1e-6:
                            return None
                            
                        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
                        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
                        
                        # 对于射线，我们只检查u的范围（是否在边上）
                        if 0 <= u <= 1:
                            x = x1 + t * (x2 - x1)
                            y = y1 + t * (y2 - y1)
                            if t >= 0:  # 确保是射线方向
                                return (int(x), int(y))
                        return None

                    # 查找射线与四边形边的交点
                    for i in range(4):
                        start = midpoints_int[i]
                        end = midpoints_int[(i + 1) % 4]
                        
                        intersection = check_intersection(
                            center, ray_end,  # 射线
                            start, end        # 四边形的边
                        )
                        
                        if intersection:
                            self.target_point = intersection
                            print(f"Angle: {self.angle:.1f}°")
                            print(f"Target: ({intersection[0]}, {intersection[1]})")
                            break

            time.sleep(0.03)  # ~30fps

def mouse_callback(event, x, y, flags, param):
    processor = param
    if not processor.is_setting_perspective:
        return
        
    if event == cv2.EVENT_LBUTTONDOWN and len(processor.perspective_points) < 4:
        processor.perspective_points.append([x, y])
        # Draw point and lines
        frame_copy = processor.frame.copy()
        for point in processor.perspective_points:
            cv2.circle(frame_copy, tuple(point), 5, (0, 255, 0), -1)
        if len(processor.perspective_points) > 1:
            for i in range(len(processor.perspective_points)-1):
                pt1 = tuple(processor.perspective_points[i])
                pt2 = tuple(processor.perspective_points[i+1])
                cv2.line(frame_copy, pt1, pt2, (0, 255, 0), 2)
        if len(processor.perspective_points) == 4:
            cv2.line(frame_copy, tuple(processor.perspective_points[-1]), 
                    tuple(processor.perspective_points[0]), (0, 255, 0), 2)
            # Calculate perspective transform
            dst_points = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
            src_points = np.float32(processor.perspective_points)
            processor.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            processor.is_setting_perspective = False
            
        processor.display_frame = frame_copy

def main():
    cap = cv2.VideoCapture(2)
    processor = FrameProcessor()

    # Create HSV control window and trackbars
    cv2.namedWindow('HSV Controls')
    cv2.createTrackbar('H_min', 'HSV Controls', 0, 180, nothing)
    cv2.createTrackbar('S_min', 'HSV Controls', 0, 255, nothing)
    cv2.createTrackbar('V_min', 'HSV Controls', 200, 255, nothing)
    cv2.createTrackbar('H_max', 'HSV Controls', 180, 180, nothing)
    cv2.createTrackbar('S_max', 'HSV Controls', 30, 255, nothing)
    cv2.createTrackbar('V_max', 'HSV Controls', 255, 255, nothing)

    # Start all processing threads
    corner_thread = threading.Thread(target=processor.process_corners)
    bright_thread = threading.Thread(target=processor.process_bright_spot)
    draw_thread = threading.Thread(target=processor.draw_frame)
    print_thread = threading.Thread(target=processor.print_coordinates)
    
    corner_thread.start()
    bright_thread.start()
    draw_thread.start()
    print_thread.start()

    # Set up mouse callback
    cv2.namedWindow('Camera Feed')
    cv2.setMouseCallback('Camera Feed', mouse_callback, processor)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                # Pause processing and update single frame
                processor.paused = True
                processor.is_setting_perspective = True
                processor.perspective_points = []
                processor.perspective_matrix = None
                processor.display_frame = frame.copy()  # Use clean frame
                print("请按顺序点击四个角点（左上、右上、右下、左下）")
            elif key == ord('r'):
                processor.perspective_matrix = None
                processor.perspective_points = []
                processor.paused = False  # Resume processing
                print("已重置透视变换")

            if processor.is_setting_perspective:
                # Keep using the same frame while setting points
                processor.frame = frame
            else:
                # Normal update when not setting points
                processor.update_frame(frame)
                processor.paused = False  # Ensure processing is resumed
            
            display = processor.display_frame if processor.display_frame is not None else frame
            cv2.imshow('Camera Feed', display)
            
            if key == ord('q'):
                break
    finally:
        processor.running = False
        corner_thread.join()
        bright_thread.join()
        draw_thread.join()
        print_thread.join()
        cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()