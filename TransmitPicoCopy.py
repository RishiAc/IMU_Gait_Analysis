import network
import utime
import socket
import machine
led = machine.Pin("LED", machine.Pin.OUT)

def connect():
    #Connect to WLAN
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect("OurHome", "7323400688")
    while wlan.isconnected() == False:
        print('Waiting for connection...')
        utime.sleep(1)
        led.off()
        utime.sleep(.25)
        led.on()
    print(wlan.ifconfig())
    led.on()
    
connect()
clientmsg = 1
send = clientmsg
serverAddress=('10.0.0.14',2222)
bufferSize = 1024
UDPClient = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

packetCount = 0
while True:
    send = str(clientmsg)
    bytesToSend = send.encode('utf-8')

    UDPClient.sendto(bytesToSend, serverAddress)
    packetCount +=1
    utime.sleep(1)
    clientmsg+=1


#####################testIMUsend################################

import network
import utime
import socket
import machine
import utime
from machine import I2C, Pin
from mpu9250 import MPU9250
led = machine.Pin("LED", machine.Pin.OUT)

def connect():
    #Connect to WLAN
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect("OurHome", "7323400688")
    while wlan.isconnected() == False:
        print('Waiting for connection...')
        utime.sleep(1)
        led.off()
        utime.sleep(.25)
        led.on()
    print(wlan.ifconfig())
    led.on()
    
    
i2c = I2C (0, sda = Pin(0), scl = Pin (1), freq = 400000)
sensor = MPU9250(i2c)

print("MPU9250 id: " + hex (sensor.whoami))


connect()
clientmsg = sensor.acceleration
send = clientmsg
serverAddress=('10.0.0.19',2222)
bufferSize = 1024
UDPClient = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


while True:
    send = str(clientmsg)
    bytesToSend = send.encode('utf-8')

    UDPClient.sendto(bytesToSend, serverAddress)
    utime.sleep(1)




import network
import utime
import socket
import machine
from simple import SIMPLE
led = machine.Pin("LED", machine.Pin.OUT)

def connect():
    #Connect to WLAN
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect("OurHome", "7323400688")
    while wlan.isconnected() == False:
        print('Waiting for connection...')
        utime.sleep(1)
        led.off()
        utime.sleep(.25)
        led.on()
    print(wlan.ifconfig())
    led.on()
    
connect()
 
send = clientmsg
serverAddress=('10.0.0.19',2222)
bufferSize = 1024
UDPClient = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


while True:
    f_x_accel = get_reading()
    clientmsg = f_x_accel
    send = str(clientmsg)
    bytesToSend = send.encode('utf-8')

    UDPClient.sendto(bytesToSend, serverAddress)
    utime.sleep(1)
