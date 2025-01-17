import socket
import pandas as pd

bufferSize = 1024
ServerPort = 2222
ServerIP = '10.0.0.11'

RPIsocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
RPIsocket.bind((ServerIP, ServerPort))#changed ServerIP to ''

print('Server is running')
message, address = RPIsocket.recvfrom(bufferSize)
#ser = serial.Serial('COM5', 9600, timeout=0.050)
#ser.open()2

#df = pd.read_csv(r"/Users/rishichudasama/Gait_IMU_Project/IMU_Gait_Analysis/DataCSVs/CNNData.csv")  # Ensure CSV contains image paths and multi-labels
global df
try:
    df = pd.read_csv(r"/Users/rishichudasama/Gait_IMU_Project/IMU_Gait_Analysis/CNNData.csv")
except FileNotFoundError:
    data = {'Ax': [0], 'Ay': [0], 'Az': [0], 'pitch': [0], 'roll': [0], 'label': [0]}
    df = pd.DataFrame(data)
    df.to_csv('CNNData.csv', index=False)
 


def receive(Dtype):

    global df
    message, address = RPIsocket.recvfrom(bufferSize)

    message = message.decode('utf-8')
    #print(message)
    array1 = message.split("#")
    #print(recvPacket)
   
    new_row_df = pd.DataFrame([{"Ax": array1[0], "Ay": array1[1], "Az": array1[2],"pitch": array1[3],"roll": array1[4], "label": Dtype}])
    df = pd.concat([df, new_row_df], ignore_index=True)
    
    df.to_csv(r"/Users/rishichudasama/Gait_IMU_Project/IMU_Gait_Analysis/CNNData.csv", index=False, encoding='utf-8')

if __name__ == '__main__':
    print('Which data collection? Enter 1 for ground, 2 for stairs. ')
    x = int(input())

    if (x == 0):
        try:
            while True:
                receive(0)
                
        except KeyboardInterrupt:
            print("\nCollection Terminated")
    elif (x == 1):
        try:
            while True:
                receive(1)
        except KeyboardInterrupt:
            print("\nCollection Terminated")
    
    
 