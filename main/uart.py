import serial
import struct

class UARTHandler:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        try:
            self.serial = serial.Serial(port, baudrate)
            print(f"UART initialized on {port} at {baudrate} baud")
        except serial.SerialException as e:
            print(f"Failed to open serial port: {e}")
            self.serial = None

    def send_forces(self, force_x, force_y):
        if self.serial is None:
            return
            
        try:
            # Pack forces as floats into bytes (8 bytes total)
            data = struct.pack('ff', force_x, force_y)
            self.serial.write(data)
        except Exception as e:
            print(f"Failed to send forces: {e}")

    def close(self):
        if self.serial:
            self.serial.close()
