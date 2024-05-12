import pyxdf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
streams, _ = pyxdf.load_xdf(r"C:\Users\local_user\Downloads\TestForProject\TimeTesting\sub-P004\ses-S001\eeg\sub-P004_ses-S001_task-Default_run-001_eeg.xdf",synchronize_clocks = True)
#creating individual arrays for each data stream



x_accel = []
y_accel = []
z_accel = []
pitch = []
roll = []
timeUs = []
packetCount = []
x = 0
#attaching data streams to arrays
for stream in streams:
    y = stream['time_series']
    for i in y:
        x_accel.append(i[0])
        y_accel.append(i[1])
        z_accel.append(i[2])
        pitch.append(i[3])
        roll.append(i[4])
        packetCount.append(i[5])
        timeUs.append(i[6])



y = stream['time_series']
plt.plot(stream['time_stamps']-stream['time_stamps'][0], x_accel)
plt.plot(stream['time_stamps']-stream['time_stamps'][0], y_accel)
plt.plot(stream['time_stamps']-stream['time_stamps'][0], z_accel)
#plt.plot(stream['time_stamps']-stream['time_stamps'][0], pitch) //weird one
#plt.plot(stream['time_stamps']-stream['time_stamps'][0], roll)


ogData = {'time': stream['time_stamps'] - stream['time_stamps'][0],'x_accel': x_accel, 'y_accel': y_accel, 'z_accel': z_accel,  'pitch':pitch, 'roll': roll, 'timeUs': timeUs, 'packet' : packetCount} #needs LabelArray
ogDf = pd.DataFrame(ogData)

ogDf.to_csv('ogOut.csv')   
"""for i in range (len(packetCount)):
    newArr = []
    diffPacket = int(packetCount[i+1]-packetCount[i])
    print(diffPacket)
    tempi = i
    clone = diffPacket
    if clone != 1 or clone != -254:
        
        for x in range (clone-1):
            packetCount.insert(tempi+1, "None")
            #tempi +=1
        i += clone"""


ogArrays = [x_accel, y_accel, z_accel, pitch, roll, packetCount,timeUs]

newArr = []
for i in range (len(packetCount)):
    newArr.append(packetCount[i])
    if i<len(packetCount) - 1:
        diff = int(packetCount[i+1]-packetCount[i])
        if diff>1:
            for j in range (diff-1):
                newArr.append(None) 

NoneIndexes = []
for x in range(len(newArr)):
    if newArr[x] == None:
        NoneIndexes.append(x)
#print(NoneIndexes) 
#print(newArr)

for array in ogArrays:
    for indexthing in NoneIndexes:
        array.insert(indexthing, None)


#making dataframe, sending out so can be used in ML
#would add LabelArray to this for future reference
time = stream['time_stamps']
data = {'x_accel': x_accel, 'y_accel': y_accel, 'z_accel': z_accel,  'pitch':pitch, 'roll': roll, 'PicotimeUs': timeUs, 'packet' : packetCount} #needs LabelArray
df = pd.DataFrame(data)

print(df)
df.to_csv('out.csv')   


newDf = pd.read_csv('out.csv')

interpdf = newDf.interpolate(method = 'linear',limit_direction ='forward')

print(interpdf)
interpdf.to_csv('interp.csv')   

plt.show()



