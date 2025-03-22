import pygame
import sys
import time
from uart import UARTHandler  # Add UART import

class XboxController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        # Wait for controller
        while pygame.joystick.get_count() == 0:
            print("Waiting for controller... Please connect Xbox controller.")
            time.sleep(1)
            pygame.joystick.quit()
            pygame.joystick.init()
            
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()
        self.uart = UARTHandler()  # Add UART handler
        
    def read_input(self):
        pygame.event.pump()
        
        # Read axis values
        left_x = round(self.controller.get_axis(0), 2)
        left_y = round(self.controller.get_axis(1), 2)
        right_x = round(self.controller.get_axis(4), 2)
        right_y = round(self.controller.get_axis(3), 2)
        
        # Convert right stick values to forces (-100 to 100)
        force_x = int(right_x * 100)
        force_y = int(right_y * 100)
        
        # Send forces through UART
        self.uart.send_forces(force_x, force_y)
        
        # Read trigger values
        lt = round((self.controller.get_axis(2) + 1) / 2, 2)
        rt = round((self.controller.get_axis(5) + 1) / 2, 2)
        
        # Read button states
        a = self.controller.get_button(0)
        b = self.controller.get_button(1)
        x = self.controller.get_button(2)
        y = self.controller.get_button(3)
        
        return {
            'left_stick': (left_x, left_y),
            'right_stick': (right_x, right_y),
            'triggers': (lt, rt),
            'buttons': {
                'A': a,
                'B': b,
                'X': x,
                'Y': y
            }
        }

    def __del__(self):  # Add destructor
        if hasattr(self, 'uart'):
            self.uart.close()

if __name__ == "__main__":
    try:
        controller = XboxController()
        print("Xbox controller connected!")
        
        while True:
            data = controller.read_input()
            print(f"\rSticks: L{data['left_stick']}, R{data['right_stick']} | "
                  f"Forces: X:{int(data['right_stick'][0]*100)}, Y:{int(data['right_stick'][1]*100)} | "
                  f"Triggers: LT:{data['triggers'][0]:.2f}, RT:{data['triggers'][1]:.2f} | "
                  f"Buttons: {data['buttons']}", end='')
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nExiting...")
        pygame.quit()
        sys.exit()
    except Exception as e:
        print(f"\nError: {e}")
        pygame.quit()
        sys.exit()
