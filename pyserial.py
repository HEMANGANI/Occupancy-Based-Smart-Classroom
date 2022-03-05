
from time import sleep
import serial
ser = serial.Serial('/dev/ttyACM0')  # open serial port
num1 = '1'
num2 = '0'
num3 = '1'
num4 = '0'
#time.sleep(2)

while True:
    s = str(num1) + '\t' + str(num2) + '\t' + str(num3) + '\t' + str(num4) + '\n'
    ser.write(s.encode())