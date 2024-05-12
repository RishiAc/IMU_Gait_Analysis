
import time
import socket
#import machine
import serial
bufferSize = 1024
from time import sleep
ServerPort = 2222
ServerIP = '10.0.0.14'
import pylsl


info = pylsl.StreamInfo("ArduinoStream", "SensorData", 1, 100, pylsl.cf_float32, "myuid12345")
outlet = pylsl.StreamOutlet(info)
message = 1
while True:
    print(message)
    #ser.write(message)
    outlet.push_sample([float(message)])
    message +=1
    time.sleep(1)