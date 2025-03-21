import pygame
import sys
import time

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
        
    def read_input(self):
        pygame.event.pump()
        
        # Read axis values
        left_x = round(self.controller.get_axis(0), 2)
        left_y = round(self.controller.get_axis(1), 2)
        right_x = round(self.controller.get_axis(4), 2)
        right_y = round(self.controller.get_axis(3), 2)
        
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

if __name__ == "__main__":
    try:
        controller = XboxController()
        print("Xbox controller connected!")
        
        while True:
            data = controller.read_input()
            print(f"\rSticks: L{data['left_stick']}, R{data['right_stick']} | "
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
