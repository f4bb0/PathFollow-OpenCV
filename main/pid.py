import time
import numpy as np

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limits=(-100, 100)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.last_error = np.array([0.0, 0.0])  # [x, y]
        self.integral = np.array([0.0, 0.0])    # [x, y]
        self.last_time = time.time()

    def reset(self):
        """Reset the PID controller"""
        self.last_error = np.array([0.0, 0.0])
        self.integral = np.array([0.0, 0.0])
        self.last_time = time.time()

    def compute(self, target_point, current_point):
        """
        Compute PID control values for both X and Y axes
        
        Args:
            target_point: tuple (x, y) - target position
            current_point: tuple (x, y) - current position
        
        Returns:
            tuple (force_x, force_y) - control forces in range (-100, 100)
        """
        if target_point is None or current_point is None:
            return 0.0, 0.0

        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            return 0.0, 0.0

        # Calculate error vector
        error = np.array([
            target_point[0] - current_point[0],
            target_point[1] - current_point[1]
        ])

        # Calculate derivative
        derivative = (error - self.last_error) / dt

        # Update integral
        self.integral += error * dt

        # Calculate output for both axes
        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        # Update state
        self.last_error = error
        self.last_time = current_time

        # Clip outputs to limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        return output[0], output[1]  # return force_x, force_y
