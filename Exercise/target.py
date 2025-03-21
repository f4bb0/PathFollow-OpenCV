import cv2
import numpy as np

def main():
    # Initialize video capture (0 is usually the built-in webcam)
    cap = cv2.VideoCapture(2)

    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Perform Canny edge detection
        edges = cv2.Canny(blurred, 
                         threshold1=100,  # Lower threshold
                         threshold2=200)  # Upper threshold

        # Show original and edge detection results
        cv2.imshow('Original', frame)
        cv2.imshow('Edges', edges)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()