"""
Make predictions on new Bitcoin price data using the trained model.
"""

import torch
import torch.nn as nn
from Network import TFN
from data import load_data_from_csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuration
MODEL_PATH = "model_200000.pt"
CSV_PATH = "data/btcusd_2012-01-01_to_2024-11-23_1min_updated_20250528.csv"

# Model parameters (must match training)
n_variables_past_continuous = 4
n_variables_future_continuous = 0
n_variables_past_discrete = [24]
n_variables_future_discrete = [24]
dim_model = 160
n_lstm_layers = 4
n_attention_layers = 3
n_heads = 6
dropout_rate = 0.2

# Sequence lengths
past_seq_len = 80  # Need 80 minutes of historical data
future_seq_len = 15  # Predict 15 minutes ahead

# Columns
continuous_columns = ['Open', 'High', 'Low', 'Close']
discrete_columns = ['Hour']
target_columns = ['Close']

# Device
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
quantiles = torch.tensor([0.1, 0.5, 0.9]).float().to(device)

print("="*60)
print("Bitcoin Price Prediction using Temporal Fusion Transformer")
print("="*60)

# Load model
print("\nLoading model...")
model = TFN(
    n_variables_past_continuous,
    n_variables_future_continuous,
    n_variables_past_discrete,
    n_variables_future_discrete,
    dim_model,
    n_quantiles=quantiles.shape[0],
    dropout_r=dropout_rate,
    n_attention_layers=n_attention_layers,
    n_lstm_layers=n_lstm_layers,
    n_heads=n_heads
).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✓ Loaded model from {MODEL_PATH}")

# Load training data for normalization
print("\nLoading data for normalization...")
full_data = load_data_from_csv(CSV_PATH)
train_data = full_data[(full_data['timestamp'] >= '2018-01-01') & 
                        (full_data['timestamp'] < '2020-01-01')].copy()

# Calculate normalization statistics
mean = train_data[continuous_columns].stack().mean()
std = train_data[continuous_columns].stack().std()
print(f"✓ Normalization stats: mean={mean:.2f}, std={std:.2f}")

def normalize_data(data, mean, std):
    """Normalize data using training statistics"""
    return (data - mean) / std

def denormalize_data(data, mean, std):
    """Denormalize data back to original scale"""
    return data * std + mean

def prepare_prediction_data(data, last_n_minutes=80):
    """
    Prepare the last N minutes of data for prediction.
    
    Args:
        data: DataFrame with timestamp, Open, High, Low, Close, Hour columns
        last_n_minutes: Number of minutes to use for prediction (default: 80)
    
    Returns:
        Prepared tensors for model input
    """
    # Get last N minutes
    recent_data = data.tail(last_n_minutes).copy()
    
    if len(recent_data) < last_n_minutes:
        raise ValueError(f"Not enough data. Need {last_n_minutes} minutes, got {len(recent_data)}")
    
    # Normalize continuous features
    recent_data[continuous_columns] = normalize_data(recent_data[continuous_columns], mean, std)
    
    # Extract features
    continuous = recent_data[continuous_columns].values  # Shape: (80, 4)
    discrete = recent_data[discrete_columns].values  # Shape: (80, 1)
    
    # Convert to tensors
    continuous_tensor = torch.tensor(continuous, dtype=torch.float32, device=device)
    discrete_tensor = torch.tensor(discrete, dtype=torch.float32, device=device)
    
    # Reshape for model: (batch=1, seq_len, n_vars, 1) for continuous
    continuous_tensor = continuous_tensor.unsqueeze(0).unsqueeze(-1)  # (1, 80, 4, 1)
    discrete_tensor = discrete_tensor.unsqueeze(0)  # (1, 80, 1)
    
    # Future discrete features (hours for next 15 minutes)
    last_timestamp = recent_data['timestamp'].iloc[-1]
    future_hours = []
    for i in range(1, future_seq_len + 1):
        future_time = last_timestamp + timedelta(minutes=i)
        future_hours.append([future_time.hour])
    
    future_discrete = torch.tensor(future_hours, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 15, 1)
    
    return continuous_tensor, discrete_tensor, future_discrete

def one_hot_encode(x, dims, device='cpu'):
    """One-hot encode discrete variables"""
    from data import one_hot
    return one_hot(x, dims, device=device)

def make_prediction(data, last_n_minutes=80):
    """
    Make predictions for the next 15 minutes.
    
    Args:
        data: DataFrame with historical price data
        last_n_minutes: Number of minutes to use for prediction
    
    Returns:
        Dictionary with predictions and metadata
    """
    # Prepare data
    continuous, discrete, future_discrete = prepare_prediction_data(data, last_n_minutes)
    
    # One-hot encode discrete variables
    discrete_encoded = one_hot_encode(discrete, n_variables_past_discrete, device=device)
    future_discrete_encoded = one_hot_encode(future_discrete, n_variables_future_discrete, device=device)
    
    # Reset model state
    model.reset(batch_size=1, device=device)
    
    # Make prediction
    with torch.no_grad():
        predictions, vs_weights = model(continuous, discrete_encoded, None, future_discrete_encoded)
    
    # Convert to numpy
    predictions = predictions.cpu().numpy()[0]  # Shape: (15, 3) - 3 quantiles
    
    # Denormalize predictions
    predictions_denorm = denormalize_data(predictions, mean, std)
    
    # Get variable weights
    vs_weights_np = torch.mean(torch.mean(vs_weights, dim=0), dim=0).squeeze().cpu().numpy()
    
    return {
        'predictions': predictions_denorm,  # Shape: (15, 3) - [10th, 50th, 90th quantiles]
        'predictions_normalized': predictions,
        'variable_weights': vs_weights_np,
        'last_timestamp': data['timestamp'].iloc[-1]
    }

# Example: Predict on recent data
print("\n" + "="*60)
print("Making predictions on recent test data...")
print("="*60)

# Get recent test data
test_data = full_data[(full_data['timestamp'] >= '2020-01-01') & 
                       (full_data['timestamp'] < '2021-01-01')].copy()

# Use last 80 minutes of test data
prediction_data = test_data.tail(80 + future_seq_len).copy()

# Get historical data for prediction
historical = prediction_data.head(80)

# Make prediction
pred_result = make_prediction(historical)

# Get actual values for comparison
actual_values = prediction_data.tail(future_seq_len)['Close'].values

print(f"\nLast historical timestamp: {pred_result['last_timestamp']}")
print(f"\nPredictions for next {future_seq_len} minutes:")
print(f"{'Minute':<8} {'10th %ile':<12} {'50th %ile (Median)':<20} {'90th %ile':<12} {'Actual':<12}")
print("-" * 70)

for i in range(future_seq_len):
    pred_10 = pred_result['predictions'][i, 0]
    pred_50 = pred_result['predictions'][i, 1]
    pred_90 = pred_result['predictions'][i, 2]
    actual = actual_values[i]
    
    print(f"{i+1:<8} ${pred_10:>10.2f}  ${pred_50:>18.2f}  ${pred_90:>10.2f}  ${actual:>10.2f}")

print(f"\nVariable Importance:")
variable_names = ['Open', 'High', 'Low', 'Close', 'Hour']
total_weight = pred_result['variable_weights'].sum()
for name, weight in zip(variable_names, pred_result['variable_weights']):
    print(f"  {name:8s}: {weight:.4f} ({100*weight/total_weight:.2f}%)")

print("\n" + "="*60)
print("Prediction complete!")
print("="*60)

# Save predictions to CSV
output_df = pd.DataFrame({
    'minute_ahead': range(1, future_seq_len + 1),
    'prediction_10th_percentile': pred_result['predictions'][:, 0],
    'prediction_median': pred_result['predictions'][:, 1],
    'prediction_90th_percentile': pred_result['predictions'][:, 2],
    'actual': actual_values
})

output_df.to_csv('predictions.csv', index=False)
print(f"\n✓ Predictions saved to: predictions.csv")

