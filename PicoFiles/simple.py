# Simply.py - a simple MPU9250 demo app
# Kevin McAleer
# May 2021

from machine import I2C, Pin
from math import sqrt, atan2, pi, copysign, sin, cos
from mpu9250 import MPU9250
from time import sleep
import utime
# addresses 
MPU = 0x68
id = 1
sda = Pin(6)
scl = Pin(7)

# create the I2C
i2c = I2C(id=id, scl=scl, sda=sda)

# Scan the bus
print(i2c.scan())
m = MPU9250(i2c)

# Calibration and bias offset
#m.ak8963.calibrate(count=100)
pitch_bias = 0.0
roll_bias = 0.0

# For low pass filtering
filtered_x_value = 0.0 
filtered_y_value = 0.0
f_x_accel = 0.0
f_y_accel = 0.0
f_z_accel = 0.0
f_pitch = 0.0
declination = 40

def degrees_to_heading(degrees):
    heading = ""
    if (degrees > 337) or (degrees >= 0 and degrees <= 22):
            heading = 'N'
    if degrees >22 and degrees <= 67:
        heading = "NE"
    if degrees >67 and degrees <= 112:
        heading = "E"
    if degrees >112 and degrees <= 157:
        heading = "SE"
    if degrees > 157 and degrees <= 202:
        heading = "S"
    if degrees > 202 and degrees <= 247:
        heading = "SW"
    if degrees > 247 and degrees <= 292:
        heading = "W"
    if degrees > 292 and degrees <= 337:
        heading = "NW"
    return heading

def get_reading()->float:
    ''' Returns the readings from the sensor '''
    global filtered_y_value, filtered_x_value, f_x_accel, f_y_accel, f_z_accel, f_pitch
    x = m.acceleration[0] 
    y = m.acceleration[1]
    z = m.acceleration[2] 

    # Pitch and Roll in Radians
    roll_rad = atan2(-x, sqrt((z*z)+(y*y)))
    pitch_rad = atan2(z, copysign(y,y)*sqrt((0.01*x*x)+(y*y)))

    # Pitch and Roll in Degrees
    pitch = pitch_rad*180/pi
    roll = roll_rad*180/pi

    # Get soft_iron adjusted values from the magnetometer
    mag_x, mag_y, magz = m.magnetic

    filtered_x_value = low_pass_filter(mag_x, filtered_x_value)
    filtered_y_value = low_pass_filter(mag_y, filtered_y_value)
    f_x_accel = low_pass_filter(x, f_x_accel)
    f_y_accel = low_pass_filter(y, f_y_accel)
    f_z_accel = low_pass_filter(z, f_z_accel)
    f_pitch = low_pass_filter(pitch, f_pitch)
    az =  90 - atan2(filtered_y_value, filtered_x_value) * 180 / pi

    # make sure the angle is always positive, and between 0 and 360 degrees
    if az < 0:
        az += 360
        
    # Adjust for original bias
    pitch -= pitch_bias
    roll -= roll_bias

    heading = degrees_to_heading(az)

    return f_x_accel, f_y_accel, f_z_accel, f_pitch, roll, az, heading

def low_pass_filter(raw_value:float, remembered_value):
    ''' Only applied 20% of the raw value to the filtered value '''
    
    # global filtered_value
    alpha = 0.1
    filtered = 0
    filtered = (alpha * remembered_value) + (1.0 - alpha) * raw_value
    return filtered

def show():
    ''' Shows the Pitch, Rool and heading '''
    f_x_accel, f_y_accel, f_z_accel, f_pitch, roll, az, heading_value = get_reading()
    #print("Pitch: ", round (f_pitch,1))
    #print("Pitch: ",round(f_pitch,1), "Roll: ",round(roll, 1), "angle: ", round(az))#,"Heading", heading_value)
    #print("x: ", f_x_accel, "y: ", f_y_accel, "z: ",f_z_accel, "pitch: ", f_pitch, "roll: ", roll)
    #print("roll: ", roll, "pitch: ",f_pitch)
    sleep(0.005)

# reset orientation to zero
f_x_accel, f_y_accel, f_z_accel, f_pitch, roll, az, heading_value = get_reading()

#main loop
# while True:
#      show()
