import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt

# Step 1: Download historical stock data using yfinance
ticker = 'AAPL'  # You can change this to any stock symbol
data = yf.download(ticker, start='2017-01-01', end='2022-12-31')
# Select relevant features: Open, High, Low, Close, Volume
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Step 2: Scale the data to range [0,1] for better LSTM performance
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Step 3: Prepare dataset sequences for LSTM input
SEQ_LENGTH = 60  # Number of days to look back for prediction

class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        # Total sequences = total data points - sequence length
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Input sequence: seq_length days of features
        x = self.data[idx:idx+self.seq_length]
        # Target: Close price of the day after the sequence
        y = self.data[idx+self.seq_length, 3]  # Close price index is 3
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Create dataset instance
dataset = StockDataset(scaled_data, SEQ_LENGTH)

# Step 4: Split dataset into training and testing sets (80% train, 20% test)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# DataLoader for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 5: Define the LSTM model architecture
class StockLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super(StockLSTM, self).__init__()
        # LSTM layer(s)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer to output a single value (predicted price)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Pass input through LSTM layers
        out, _ = self.lstm(x)
        # Take output from the last time step
        out = out[:, -1, :]
        # Pass through fully connected layer
        out = self.fc(out)
        return out.squeeze()  # Remove extra dimensions

# Instantiate model and move to GPU if available
model = StockLSTM()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Step 6: Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

EPOCHS = 30  # Number of training epochs

# Step 7: Training loop
for epoch in range(EPOCHS):
    model.train()  # Set model to training mode
    train_losses = []
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # Clear gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        train_losses.append(loss.item())
    avg_loss = np.mean(train_losses)
    print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {avg_loss:.6f}")

# Step 8: Evaluation on test data
model.eval()  # Set model to evaluation mode
predictions = []
actuals = []
with torch.no_grad():  # Disable gradient calculation
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(targets.numpy())

# Convert lists to numpy arrays and reshape for inverse scaling
predictions = np.array(predictions).reshape(-1, 1)
actuals = np.array(actuals).reshape(-1, 1)

# Step 9: Inverse transform the scaled data to original price scale
# Since scaler was fit on all features, create dummy arrays to inverse transform only Close price
dummy_pred = np.zeros((len(predictions), scaled_data.shape[1]))
dummy_pred[:, 3] = predictions[:, 0]  # Close price column

dummy_actual = np.zeros((len(actuals), scaled_data.shape[1]))
dummy_actual[:, 3] = actuals[:, 0]

predicted_prices = scaler.inverse_transform(dummy_pred)[:, 3]
actual_prices = scaler.inverse_transform(dummy_actual)[:, 3]

# Step 10: Calculate Root Mean Squared Error (RMSE) as performance metric
rmse = math.sqrt(mean_squared_error(actual_prices, predicted_prices))
print(f"Test RMSE: {rmse:.4f}")

# Step 11: Plot actual vs predicted prices for visual comparison
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.title(f'{ticker} Stock Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
