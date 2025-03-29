import serial
import struct
import time
class UARTHandler:
    def __init__(self,port='/dev/ttyUSB0', baudrate=115200, bytesize=serial.EIGHTBITS,    parity=serial.PARITY_NONE,  stopbits=serial.STOPBITS_ONE):
        try:
            self.serial = serial.Serial(port, baudrate, bytesize,  parity,stopbits)
            print(f"UART initialized on {port} at {baudrate} baud")
        except serial.SerialException as e:
            print(f"Failed to open serial port: {e}")
            self.serial = None

    def send_forces(self, force_x, force_y):
        if self.serial is None:
            return
            
        try:
            # Send X force to address 1
            vel_x = int(abs(force_x/10))
            dir_x = 1 if force_x >= 0 else 0
            self.Emm_V5_Vel_Control(addr=1, dir=dir_x, vel=vel_x, acc=50, snF=False)
            
            # Send Y force to address 2
            vel_y = int(abs(force_y/10))
            dir_y = 1 if force_y >= 0 else 0
            self.Emm_V5_Vel_Control(addr=2, dir=dir_y, vel=vel_y, acc=50, snF=False)
        except Exception as e:
            print(f"Failed to send forces: {e}")

    def Emm_V5_Vel_Control(self, addr, dir, vel, acc, snF):
        if self.serial is None:
            return
            
        try:
            cmd = bytearray(16)
            cmd[0] = addr                  # 地址
            cmd[1] = 0xF6                  # 功能码
            cmd[2] = dir                   # 方向，0为CW，其余值为CCW9
            cmd[3] = (vel >> 8) & 0xFF     # 速度(RPM)高8位字节
            cmd[4] = vel & 0xFF            # 速度(RPM)低8位字节
            cmd[5] = acc                  # 加速度，注意：0是直接启动
            cmd[6] = 0x01 if snF else 0x00 # 多机同步运动标志
            cmd[7] = 0x6B                  # 校验字节
            self.serial.write(cmd[:8])
            time.sleep(0.01)
        except Exception as e:
            print(f"Failed to send motor control command: {e}")

    def close(self):
        if self.serial:
            self.serial.close()
