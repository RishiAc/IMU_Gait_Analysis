import socket
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import joblib
from collections import deque

bufferSize = 1024
ServerPort = 2222
ServerIP = '10.0.0.11'

RPIsocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
RPIsocket.bind((ServerIP, ServerPort))#changed ServerIP to ''

print('Server is running')



class CNNModel(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1)  # padding=1 to preserve length
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)  # padding=1 to preserve length

        self.dropout = nn.Dropout(p=0.2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * (n_timesteps // 2), 100)
        self.fc2 = nn.Linear(100, n_outputs)
        
    def forward(self, x):
        #if len(x.shape) == 3:  # Ensure it's 3D before transposing
        x = x.transpose(1, 2)  # Change to (batch_size, num_features, time_steps)
        # elif len(x.shape) == 2:  # If it's 2D, add an additional dimension
        #     x = x.unsqueeze(1)  # Add a dummy time dimension if missing
        #     x = x.transpose(1, 2)
        
        x = torch.relu(self.conv1(x))
        
        x = torch.relu(self.conv2(x))
        
        x = self.dropout(x)
        
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
 


def predict(model, window_tensor):
    window_tensor = window_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(window_tensor)
        probabilities = torch.softmax(output, dim=1)

        _, predicted_classes = torch.max(output, dim=1)
        return predicted_classes.item()

def load_model(file_path,  n_timesteps, n_features, n_outputs):
    model = CNNModel( n_timesteps, n_features, n_outputs)
    model.load_state_dict(torch.load(file_path))
    model.eval()  # Set to evaluation mode
    return model
def receive():
    scaler = StandardScaler()
    X_window = []
    #X_window = [np.array([], dtype=float)]

    for i in range(120):
        message, address = RPIsocket.recvfrom(bufferSize)
        message = message.decode('utf-8')
        array1 = message.split("#") 
        float_list = [float(x) for x in array1]
        X_window.append(float_list)



    #print(X_window.shape)
    X_window = np.array(X_window)
    #X_window = X_window.unsqueeze(0)
    print(X_window)
    #X_window = X_window.transpose(1,2)  # Add batch dimension -> (1, timesteps, features)
    #print(X_window.shape)
    # Assuming the scaler was already fit during training
    scaler = joblib.load('scaler.pkl')  # Load the pre-fitted scaler
    X_window = scaler.transform(X_window.reshape(-1, X_window.shape[-1])).reshape(X_window.shape)
    X_window = torch.tensor(X_window, dtype=torch.float32)
    print(X_window.shape)
    # Convert to torch tensor
    
    return X_window

if __name__ == '__main__':
    model = load_model("/Users/rishichudasama/Gait_IMU_Project/IMU_Gait_Analysis/GaitModel.pth", 120,5,2)
    
    print('Enter 1 to start ')
    x = int(input())

    if (x == 1):
        try:
            while True:
                predictthing = receive()
                print(predictthing.shape)
                #print(predictthing)
                labels = predict (model, predictthing)
                print(labels)



                
        except KeyboardInterrupt:
            print("\nCollection Terminated")
    
    

 