import machine
import utime

led = machine.Pin("LED", machine.Pin.OUT)

while True:
    led.off()
    utime.sleep(.5)
    led.on()
    utime.sleep(.25)