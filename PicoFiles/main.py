import network
import utime
import socket
import machine
import time

from simple import get_reading
led = machine.Pin("LED", machine.Pin.OUT)

def connect():
    #Connect to WLAN
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect("cantfind", "sch0l@st1c")
    while wlan.isconnected() == False:
        print('Waiting for connection...')
        utime.sleep(1)
        led.off()
        utime.sleep(.25)
        led.on()
    print(wlan.ifconfig())
    led.on()
    
connect()
clientmsg = get_reading()[0:5]
send = clientmsg
serverAddress=('192.168.0.101',2222)
bufferSize = 1024
UDPClient = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

packetCount = 0

while (1):
    while (packetCount<255):
        timeCount = time.ticks_us()
        s = get_reading()[0:5]
        #s = 0,0,0,0,0
        packet = str(packetCount)
        timeTuple = (str(timeCount))
        s = s+(packet,timeTuple,)
        clientmsg = "#".join((map(str, s)))
        bytesToSend = clientmsg.encode('utf-8')

        UDPClient.sendto(bytesToSend, serverAddress)
        utime.sleep(.01)


        packetCount +=1
    packetCount = 0
   
    #print("PacketCOunt:"+str(packetCount))
    
    