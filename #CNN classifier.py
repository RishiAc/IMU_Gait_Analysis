import numpy as np
import socket
import pandas as pd

bufferSize = 1024
ServerPort = 2222
ServerIP = '10.0.0.31'

RPIsocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
RPIsocket.bind((ServerIP, ServerPort))#changed ServerIP to ''

print('Server is running')
message, address = RPIsocket.recvfrom(bufferSize)
#ser = serial.Serial('COM5', 9600, timeout=0.050)
#ser.open()
#df = pd.read_csv(r"C:/Users/local_user/Downloads/TestForProject/DataCSVs/CNNData.csv")  # Ensure CSV contains image paths and multi-labels

data = {'Ax': [0], 'Ay': [0], 'Az': [0], 'pitch': [0], 'roll': [0]}
df = pd.DataFrame(data)
df.to_csv('my_data.csv', index=False)

def receive():
    global df
    message, address = RPIsocket.recvfrom(bufferSize)

    message = message.decode('utf-8')
    #print(message)
    array1 = message.split("#")

    #print(recvPacket)
    new_row_df = pd.DataFrame([{"Ax": array1[0], "Ay": array1[1], "Az": array1[1],"pitch": array1[1],"roll": array1[1]}])
    df = pd.concat([df, new_row_df], ignore_index=True)
    df.to_csv(r"\Users\local_user\Downloads\TestForProject\my_data.csv", index=False, encoding='utf-8')

if __name__ == '__main__':

    receive()

    

#while True:
 #   message, address = RPIsocket.recvfrom(bufferSize)

  #  message = message.decode('utf-8')
    #print(message)
   # array1 = message.split("#")
    #my_array = np.array(array1[0], array1[1],array1[2],array1[3],array1[4])
    #print(my_array)
    #print(recvPacket)
    #"Ax","Ay","Az","pitch","roll","packetCount", "timeUs"
    #new_row_df = pd.DataFrame([{"Ax": array1[0], "Ay": array1[1], "Az": array1[1],"pitch": array1[1],"roll": array1[1]}])
    #df = pd.concat([df, new_row_df], ignore_index=True)
    #df.to_csv("\Users\local_user\Downloads\TestForProject\CNN.csv", index=False, encoding='utf-8')
