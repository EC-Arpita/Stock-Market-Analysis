import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import math
import matplotlib.pyplot as plt

# Step 1: Download historical stock data
ticker = 'AAPL'  # Stock symbol (Apple). You can change it to any ticker.
data = yf.download(ticker, start='2017-01-01', end='2022-12-31')

# Select only relevant features: Open, High, Low, Close, Volume (OHLCV)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Step 2: Scale the data (normalization)
# LSTM models perform better when inputs are normalized
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


# Step 3: Create a custom Dataset class for sequence data
SEQ_LENGTH = 60  # Number of days to look back for prediction

class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        # Number of possible sequences
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Input = past 'seq_length' days of all features
        x = self.data[idx:idx+self.seq_length]
        # Target = next day's Close price (index 3 in OHLCV)
        y = self.data[idx+self.seq_length, 3]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
# Step 4: Define the LSTM model
class StockLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super(StockLSTM, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer → predict next Close price
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # LSTM output for all time steps
        out = out[:, -1, :]            # Take only the last time step
        out = self.fc(out)             # Pass through fully connected layer
        return out.squeeze()           # Return as 1D tensor

# Step 5: Cross-validation + Hyperparameter Tuning
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training + evaluation function for one fold
def train_evaluate_lstm(train_idx, val_idx, data, seq_length, params, device):
    # Split into training & validation using fold indices
    train_data = data[train_idx]
    val_data = data[val_idx]

    # Create PyTorch datasets & loaders
    train_dataset = StockDataset(train_data, seq_length)
    val_dataset = StockDataset(val_data, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    # Initialize model
    model = StockLSTM(input_size=5,
                      hidden_size=params['hidden_size'],
                      num_layers=params['num_layers']).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # Training loop
    for epoch in range(params['epochs']):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Validation loop
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds.extend(outputs.cpu().numpy())
            actuals.extend(targets.numpy())

    # Convert predictions back to original price scale
    preds = np.array(preds).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)

    dummy_pred = np.zeros((len(preds), scaled_data.shape[1]))
    dummy_pred[:, 3] = preds[:, 0]
    dummy_actual = np.zeros((len(actuals), scaled_data.shape[1]))
    dummy_actual[:, 3] = actuals[:, 0]

    pred_prices = scaler.inverse_transform(dummy_pred)[:, 3]
    actual_prices = scaler.inverse_transform(dummy_actual)[:, 3]

    # RMSE as evaluation metric
    rmse = math.sqrt(mean_squared_error(actual_prices, pred_prices))
    return rmse

# Hyperparameter grid to test
param_grid = [
    {'hidden_size': 32, 'num_layers': 1, 'lr': 0.001, 'batch_size': 64, 'epochs': 20},
    {'hidden_size': 64, 'num_layers': 2, 'lr': 0.001, 'batch_size': 64, 'epochs': 30},
    {'hidden_size': 128, 'num_layers': 2, 'lr': 0.0005, 'batch_size': 32, 'epochs': 40},
]

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_params = None
best_rmse = float("inf")

for params in param_grid:
    fold_rmses = []
    for train_idx, val_idx in kf.split(scaled_data):
        rmse = train_evaluate_lstm(train_idx, val_idx, scaled_data, SEQ_LENGTH, params, device)
        fold_rmses.append(rmse)
    avg_rmse = np.mean(fold_rmses)
    print(f"Params: {params}, Avg RMSE: {avg_rmse:.4f}")

    if avg_rmse < best_rmse:
        best_rmse = avg_rmse
        best_params = params

print(f"\n✅ Best Hyperparameters: {best_params}")
print(f"✅ Best CV RMSE: {best_rmse:.4f}")

# Step 6: Retrain final model with best hyperparameters
dataset = StockDataset(scaled_data, SEQ_LENGTH)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

final_model = StockLSTM(input_size=5,
                        hidden_size=best_params['hidden_size'],
                        num_layers=best_params['num_layers']).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])

# Train on full training set
for epoch in range(best_params['epochs']):
    final_model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = final_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Step 7: Final Evaluation on Test Data
final_model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = final_model(inputs)
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(targets.numpy())

# Convert predictions back to original scale
predictions = np.array(predictions).reshape(-1, 1)
actuals = np.array(actuals).reshape(-1, 1)

dummy_pred = np.zeros((len(predictions), scaled_data.shape[1]))
dummy_pred[:, 3] = predictions[:, 0]
dummy_actual = np.zeros((len(actuals), scaled_data.shape[1]))
dummy_actual[:, 3] = actuals[:, 0]

predicted_prices = scaler.inverse_transform(dummy_pred)[:, 3]
actual_prices = scaler.inverse_transform(dummy_actual)[:, 3]

# Evaluation metrics
rmse = math.sqrt(mean_squared_error(actual_prices, predicted_prices))
r2 = r2_score(actual_prices, predicted_prices)
print(f"\nFinal Test RMSE: {rmse:.4f}")
print(f"Final R² Score (accuracy-style metric): {r2:.4f}")

# Step 8: Plot Actual vs Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.title(f'{ticker} Stock Price Prediction using LSTM (Best Params)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

