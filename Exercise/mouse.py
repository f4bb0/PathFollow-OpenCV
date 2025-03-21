import cv2
import numpy as np
import time  # Add this import at the top

def find_and_draw_target(frame, template):
    # Convert both images to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Create a flipped version of the template
    flipped_template = cv2.flip(gray_template, 1)  # 1 for horizontal flip
    
    # Create a list to store different scales
    scales = np.linspace(0.2, 2.0, 20)
    max_val = 0
    best_box = None
    is_flipped = False
    
    for scale in scales:
        # Check original template
        resized_template = cv2.resize(gray_template, None, fx=scale, fy=scale)
        resized_flipped = cv2.resize(flipped_template, None, fx=scale, fy=scale)
        
        if resized_template.shape[0] > gray_frame.shape[0] or \
           resized_template.shape[1] > gray_frame.shape[1]:
            continue
            
        # Apply template matching for both original and flipped
        result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, curr_max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        result_flipped = cv2.matchTemplate(gray_frame, resized_flipped, cv2.TM_CCOEFF_NORMED)
        min_val_f, curr_max_val_f, min_loc_f, max_loc_f = cv2.minMaxLoc(result_flipped)
        
        # Check which version gives better match
        if curr_max_val > max_val or curr_max_val_f > max_val:
            if curr_max_val > curr_max_val_f:
                max_val = curr_max_val
                w = int(resized_template.shape[1])
                h = int(resized_template.shape[0])
                best_box = (max_loc[0], max_loc[1], w, h)
                is_flipped = False
            else:
                max_val = curr_max_val_f
                w = int(resized_flipped.shape[1])
                h = int(resized_flipped.shape[0])
                best_box = (max_loc_f[0], max_loc_f[1], w, h)
                is_flipped = True
    
    # Draw rectangle if match found
    if max_val > 0.7 and best_box is not None:  # Threshold can be adjusted
        x, y, w, h = best_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Add confidence text and flip status
        status = "Flipped" if is_flipped else "Normal"
        confidence_text = f"Confidence: {max_val:.2f} ({status})"
        cv2.putText(frame, confidence_text, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main():
    # Load the template image
    template = cv2.imread('mouse.jpg')
    if template is None:
        print("Error: Could not load template image")
        return
        
    # Initialize video capture
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    # Initialize FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break
            
        # Process frame
        processed_frame = find_and_draw_target(frame, template)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count >= 30:  # Update FPS every 30 frames
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
        
        # Display FPS on frame
        cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        # Display result
        cv2.imshow('Target Detection', processed_frame)
        
        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()