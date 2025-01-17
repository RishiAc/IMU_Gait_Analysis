import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import joblib



class IMUDataset(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X (array-like): Features (input data).
            y (array-like): Labels.
        """
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert features to tensors
        self.y = torch.tensor(y, dtype=torch.long)  # Convert labels to tensors (for classification)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return the feature and corresponding label at the given index
        return self.X[idx], self.y[idx]





class CNNModel(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1)  # padding=1 to preserve length
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)  # padding=1 to preserve length

        self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * (n_timesteps // 2), 100)
        self.fc2 = nn.Linear(100, n_outputs)
        
    def forward(self, x):
        print(x)
        print(x.shape)
        print(x)

        if len(x.shape) == 3:  # Ensure it's 3D before transposing
            x = x.transpose(1, 2)  # Change to (batch_size, num_features, time_steps)
        elif len(x.shape) == 2:  # If it's 2D, add an additional dimension
            x = x.unsqueeze(1)  # Add a dummy time dimension if missing
            x = x.transpose(1, 2)
        print(x)
        print(x.shape)
        #print(f"Shape before conv1: {x.shape}")
        x = torch.relu(self.conv1(x))
        #print(f"Shape after conv1: {x.shape}")
        
        x = torch.relu(self.conv2(x))
        #print(f"Shape after conv2: {x.shape}")
        
        x = self.dropout(x)
        
        #print(f"Shape before pooling: {x.shape}")
        x = self.pool(x)
        #print(f"Shape after pooling: {x.shape}")
        
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        #print(f"Shape after flattening: {x.shape}")
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
 


def createWindows(X, y, window_size):
    X_windows = []
    y_windows = []
    
    # Loop over the data to create windows
    for i in range(len(X) - window_size + 1):
        # Create window of size (window_size, num_features)
        X_window = X[i:i+window_size, :]
        X_windows.append(X_window)
        
        # Use the last label in the window as the window's label
        y_window = y[i + window_size - 1]
        y_windows.append(y_window)
    
    return np.array(X_windows), np.array(y_windows)



data = pd.read_csv('/Users/rishichudasama/Gait_IMU_Project/IMU_Gait_Analysis/CNNData.csv')


X = data[['Ax', 'Ay', 'Az', 'pitch', 'roll']].values
y = data['label'].values

#Need to change 120 to actual hz of device so it's around 2 seconds. Need to edit pico hardware code to have delay, sent timestamps, so can dynamically scale it
windowSize = 120
X_windows, y_windows = createWindows(X,y, windowSize)

X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

joblib.dump(scaler, 'scaler.pkl')


train_data = IMUDataset(X_train, y_train)
test_data = IMUDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define model hyperparameters
n_timesteps = 120  # Adjust as per your input sequence length
n_features = 5   # Adjust based on the number of features (e.g., Ax, Ay, Az, pitch, roll)
n_outputs = 2      # Adjust based on the number of output classes (e.g., 0 = flat ground, 1 = stairs, etc.)

# Instantiate the model
model = CNNModel(n_timesteps, n_features, n_outputs)
criterion = nn.CrossEntropyLoss()  # For classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        #print("-------one input--------")
        #print(inputs)
        #print(inputs.shape)
        #print("-------one input--------")

        outputs = model(inputs)  # Forward pass
        probabilities = torch.softmax(outputs, dim=1)

        _, predicted_classes = torch.max(outputs, dim=1)
        #print(predicted_classes)        
        loss = criterion(outputs, labels)  # Calculate the loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update the weights

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

# Example usage
save_model(model, '/Users/rishichudasama/Gait_IMU_Project/IMU_Gait_Analysis/GaitModel.pth')