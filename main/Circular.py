import cv2
import numpy as np
import threading
import time
from pid import PIDController  # Add this import at the top
from uart import UARTHandler  # Add this import at top

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
        self.outer_ellipse = None
        self.inner_ellipse = None
        self.middle_ellipse = None
        # Remove corner-related attributes
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
        self.pid_controller = PIDController(kp=1.0, ki=0.1, kd=0.05)  # Add PID controller
        self.uart = UARTHandler()  # Add UART handler
        self.forces_locked = False  # Add this line for force lock status

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

    def normalize_angle(self, angle):
        """规范化椭圆角度，处理90度偏转问题"""
        # 确保角度在0-180范围内
        angle = angle % 180
        return angle

    def calculate_middle_ellipse(self, outer_ellipse, inner_ellipse):
        """计算中间椭圆，处理角度偏转问题"""
        outer_center, outer_axes, outer_angle = outer_ellipse
        inner_center, inner_axes, inner_angle = inner_ellipse
        
        # 计算中心点
        middle_center = ((outer_center[0] + inner_center[0])/2, 
                        (outer_center[1] + inner_center[1])/2)
        
        # 计算长短轴
        middle_axes = ((outer_axes[0] + inner_axes[0])/2,
                      (outer_axes[1] + inner_axes[1])/2)
        
        # 规范化两个角度
        outer_angle = self.normalize_angle(outer_angle)
        inner_angle = self.normalize_angle(inner_angle)
        
        # 处理角度差异
        angle_diff = abs(outer_angle - inner_angle)
        if angle_diff > 90:
            # 如果角度差大于90度，将其中一个角度旋转180度
            if outer_angle > inner_angle:
                outer_angle -= 180
            else:
                inner_angle -= 180
        
        # 计算平均角度
        middle_angle = (outer_angle + inner_angle) / 2
        # 确保最终角度在0-180范围内
        middle_angle = self.normalize_angle(middle_angle)
        
        return (middle_center, middle_axes, middle_angle)

    def detect_ellipses(self):  # renamed from process_corners
        while self.running:
            if self.paused:
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
                time.sleep(0.1)
                continue

            # Find and fit outer ellipse
            outer_contour = max(contours, key=cv2.contourArea)
            if len(outer_contour) >= 5:
                outer_ellipse = cv2.fitEllipse(outer_contour)
                
                # Create mask for inner ellipse detection
                mask = np.zeros_like(gray)
                cv2.ellipse(mask, outer_ellipse, 255, -1)
                kernel = np.ones((5,5), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
                
                # Find inner ellipse
                edges_roi = cv2.bitwise_and(edges, edges, mask=mask)
                contours_inner, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours_inner:
                    inner_contour = max(contours_inner, key=cv2.contourArea)
                    if len(inner_contour) >= 5:
                        inner_ellipse = cv2.fitEllipse(inner_contour)
                        
                        # 使用新的计算方法
                        middle_ellipse = self.calculate_middle_ellipse(outer_ellipse, inner_ellipse)
                        
                        with self.draw_lock:
                            self.outer_ellipse = outer_ellipse
                            self.inner_ellipse = inner_ellipse
                            self.middle_ellipse = middle_ellipse
            
            time.sleep(1)

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
                outer_ellipse = self.outer_ellipse
                inner_ellipse = self.inner_ellipse
                middle_ellipse = self.middle_ellipse
            
            with self.bright_center_lock:
                bright_center = self.bright_center

            # Draw using local copies
            display = frame.copy()
            if outer_ellipse is not None:
                cv2.ellipse(display, outer_ellipse, (0, 255, 0), 2)
                cv2.circle(display, (int(outer_ellipse[0][0]), int(outer_ellipse[0][1])), 5, (0, 0, 255), -1)
            if inner_ellipse is not None:
                cv2.ellipse(display, inner_ellipse, (255, 0, 0), 2)
                cv2.circle(display, (int(inner_ellipse[0][0]), int(inner_ellipse[0][1])), 5, (0, 255, 255), -1)
            if middle_ellipse is not None:
                cv2.ellipse(display, middle_ellipse, (0, 255, 255), 2)
                middle_center = (int(middle_ellipse[0][0]), int(middle_ellipse[0][1]))
                cv2.circle(display, middle_center, 5, (255, 0, 255), -1)
                
                # Draw rotating ray and target point
                if self.target_point is not None:
                    # Draw ray from center to target point
                    cv2.line(display, middle_center, self.target_point, (0, 0, 255), 2)
                    # Draw target point
                    cv2.circle(display, self.target_point, 7, (0, 0, 255), -1)
            
            if bright_center is not None:
                cv2.circle(display, bright_center, 5, (255, 0, 0), -1)
            
            with self.frame_lock:
                self.display_frame = display
                self.last_display = display
            
            time.sleep(0.03)

    def print_coordinates(self):
        """Fourth thread for calculating rotating ray intersection and PID control"""
        last_time = time.time()
#self.angular_velocity = 10  # 度/秒
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            if not self.paused and self.middle_ellipse is not None:
                # 获取中间椭圆参数
                (center_x, center_y), (a, b), angle = self.middle_ellipse
                center = (int(center_x), int(center_y))
                
                # 更新旋转角度
                self.angle = (self.angle + self.angular_velocity * dt) % 360
                angle_rad = np.radians(self.angle)
                
                # 将椭圆角度转换为弧度
                ellipse_angle_rad = np.radians(angle)
                
                # 在椭圆自身坐标系中计算
                rel_angle = angle_rad - ellipse_angle_rad
                cos_t = np.cos(rel_angle)
                sin_t = np.sin(rel_angle)
                
                # 直接计算椭圆上的点
                x = (a/2) * cos_t  # 注意这里使用 a/2 因为OpenCV的axes是直径
                y = (b/2) * sin_t  # 注意这里使用 b/2 因为OpenCV的axes是直径
                
                # 旋转回原坐标系
                rot_x = x * np.cos(ellipse_angle_rad) - y * np.sin(ellipse_angle_rad)
                rot_y = x * np.sin(ellipse_angle_rad) + y * np.cos(ellipse_angle_rad)
                
                # 平移到椭圆中心
                self.target_point = (
                    int(center_x + rot_x),
                    int(center_y + rot_y)
                )
                
                # Get current bright spot position
                with self.bright_center_lock:
                    bright_spot = self.bright_center

                # Calculate PID control if we have both target and current position
                if self.target_point is not None and bright_spot is not None:
                    if not self.forces_locked:
                        force_x, force_y = self.pid_controller.compute(
                            self.target_point, 
                            bright_spot
                        )
                    else:
                        force_x, force_y = 0.0, 0.0
                    print(f"Target: {self.target_point}")
                    print(f"Current: {bright_spot}")
                    print(f"Control forces - X: {force_x:.2f}, Y: {force_y:.2f}")
                    self.uart.send_forces(force_x, force_y)  # Add this line
                else:
                    self.pid_controller.reset()  # Reset PID when points are not available

            time.sleep(0.03)  # ~30fps

    def __del__(self):  # Add destructor
        if hasattr(self, 'uart'):
            self.uart.close()

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
    corner_thread = threading.Thread(target=processor.detect_ellipses)  # updated reference
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
            elif key == ord('s') or key == ord('S'):
                processor.forces_locked = not processor.forces_locked
                if processor.forces_locked:
                    print("Forces locked at (0, 0)")
                else:
                    print("Forces unlocked")

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