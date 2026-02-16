"""
Evaluate the trained Temporal Fusion Transformer model and generate visualizations
matching the original repository's format.
"""

# Add workspace root to path for imports
import sys
from pathlib import Path
script_dir = Path(__file__).parent
workspace_root = script_dir.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import torch
import torch.nn as nn
from Network import TFN
from data import load_data_from_csv, get_batches
from utils import forward_pass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Try to use original repository's candlestick function
try:
    from mplfinance import candlestick_ohlc
except ImportError:
    try:
        from mpl_finance import candlestick_ohlc
    except ImportError:
        print("Warning: mplfinance not available, using manual candlestick drawing")
        candlestick_ohlc = None

# Configuration
CSV_PATH = str(workspace_root / "data/btcusd_2012-01-01_to_2024-11-23_1min_updated_20250528.csv")
MODEL_PATH = str(workspace_root / "models/model_200000.pt")

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

# Sequence lengths - Match original REPOSITORY TIME DURATION
# Original repo: 80 steps * 5min = 400 minutes past, 15 steps * 5min = 75 minutes future
# With 1-minute data: 400 steps * 1min = 400 minutes past, 75 steps * 1min = 75 minutes future
# Model architecture supports variable sequence lengths, so we can use longer sequences
past_seq_len = 400  # 400 minutes = matches original 80 * 5min = 400min
future_seq_len = 75  # 75 minutes = matches original 15 * 5min = 75min

# Columns
continuous_columns = ['Open', 'High', 'Low', 'Close']
discrete_columns = ['Hour']
target_columns = ['Close']

# Device
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
quantiles = torch.tensor([0.1, 0.5, 0.9]).float().to(device)

print(f"Using device: {device}")
print("="*60)

# Load model
print("Loading model...")
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

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded model from {MODEL_PATH}")

# Load loss history
losses = checkpoint.get('losses', [])
test_losses = checkpoint.get('test_losses', [])
print(f"Training steps: {len(losses):,}")
print(f"Test evaluations: {len(test_losses):,}")

# Load data
print("\nLoading data...")
full_data = load_data_from_csv(CSV_PATH)

# Split data (same as training)
train_start_date = '2018-01-01'
train_end_date = '2020-01-01'
test_start_date = '2020-01-01'
test_end_date = '2021-01-01'

train_data = full_data[(full_data['timestamp'] >= train_start_date) & 
                        (full_data['timestamp'] < train_end_date)].copy()
test_data = full_data[(full_data['timestamp'] >= test_start_date) & 
                       (full_data['timestamp'] < test_end_date)].copy()

print(f"Training data: {len(train_data):,} rows")
print(f"Test data: {len(test_data):,} rows")

# Create data generators
test_gen = get_batches(
    test_data,
    in_seq_len=past_seq_len,
    out_seq_len=future_seq_len,
    con_cols=continuous_columns,
    disc_cols=discrete_columns,
    target_cols=target_columns,
    batch_size=4,  # Show 4 examples
    norm=train_data,
    device=device
)

print("\nGenerating predictions...")
with torch.no_grad():
    loss, net_out, vs_weights, given_data = forward_pass(
        model, test_gen, batch_size=4, quantiles=quantiles,
        discrete_dims=n_variables_past_discrete,
        device=device
    )

# Extract data
in_seq_continuous = given_data[0].cpu().numpy()
in_seq_discrete = given_data[1]
future_in_seq_discrete = given_data[2]
target_seq = given_data[3].cpu().numpy()
predictions = net_out.cpu().numpy()  # Shape: (batch, future_seq_len, 3 quantiles)

print(f"Test loss: {loss.item():.6f}")
print("="*60)

# ============================================================================
# PLOT 1: Training and Test Loss
# ============================================================================
print("\nGenerating loss plots...")
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Training loss
ax1.plot(losses[250:], label='Training Loss', linewidth=1)
ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Test loss
if len(test_losses) > 5:
    ax2.plot(test_losses[5:], label='Test Loss', linewidth=1, color='orange')
    ax2.set_title('Test Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Evaluation Step')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

plt.tight_layout()
results_dir = workspace_root / "results"
results_dir.mkdir(exist_ok=True)
plt.savefig(str(results_dir / 'training_losses.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {results_dir / 'training_losses.png'}")

# ============================================================================
# PLOT 2: Predictions vs Actual (Candlestick + Predictions)
# ============================================================================
print("\nGenerating prediction plots...")

def create_candlestick_original_format(ax, data, start_idx=0):
    """
    Create candlestick chart using original repository's exact format.
    Matches: c = torch.cat((torch.arange(-past_seq_len, 0).unsqueeze(-1).unsqueeze(-1).float(), c), dim=1)
    """
    # data shape: (seq_len, 4, 1) or (seq_len, 4)
    if len(data.shape) == 3:
        data = data.squeeze(-1)
    
    # Create x-axis values (matching original: torch.arange(-past_seq_len, 0))
    x_axis = np.arange(start_idx, start_idx + len(data))
    
    # Format: (seq_len, 5) with columns [x, Open, High, Low, Close]
    # This matches original repository's format exactly
    ohlc_data = np.column_stack([
        x_axis,
        data[:, 0],  # Open
        data[:, 1],  # High
        data[:, 2],  # Low
        data[:, 3]   # Close
    ])
    
    if candlestick_ohlc is not None:
        # Use original repository's candlestick function
        candlestick_ohlc(ax, ohlc_data, colorup="green", colordown="red", width=0.6, alpha=0.8)
    else:
        # Fallback: manual drawing
        from matplotlib.patches import Rectangle
        for i in range(len(data)):
            x = x_axis[i]
            color = 'green' if data[i, 3] >= data[i, 0] else 'red'
            ax.plot([x, x], [data[i, 2], data[i, 1]], color='black', linewidth=0.5)
            body_height = abs(data[i, 3] - data[i, 0])
            body_bottom = min(data[i, 0], data[i, 3])
            rect = Rectangle((x - 0.3, body_bottom), 0.6, body_height, 
                            facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.7)
            ax.add_patch(rect)

fig2, axes = plt.subplots(4, 1, figsize=(14, 10))

for idx in range(4):
    ax = axes[idx]
    
    # Historical data (past_seq_len steps)
    hist_data = in_seq_continuous[idx]  # Shape: (past_seq_len, 4, 1)
    
    # Create x-axis for historical data
    hist_x = np.arange(-past_seq_len, 0)
    
    # Plot candlesticks for historical data using original repository's format
    create_candlestick_original_format(ax, hist_data, start_idx=-past_seq_len)
    
    # Future predictions
    future_x = np.arange(0, future_seq_len)
    
    # Get predictions: quantiles [0.1, 0.5, 0.9]
    # NOTE: Model outputs quantiles in REVERSE order [90th, 50th, 10th]
    # We need to reverse them to get [10th, 50th, 90th]
    pred_raw = predictions[idx, :, :]  # Shape: (future_seq_len, 3)
    
    # Check if quantiles are in correct order (10th < 50th < 90th)
    # Model outputs them as [90th, 50th, 10th] so we reverse
    if pred_raw[0, 0] > pred_raw[0, 1]:  # First > second means reversed
        pred_10 = pred_raw[:, 2]  # 10th percentile (lowest - was last)
        pred_50 = pred_raw[:, 1]  # 50th percentile (median - stays middle)
        pred_90 = pred_raw[:, 0]  # 90th percentile (highest - was first)
    else:
        pred_10 = pred_raw[:, 0]  # 10th percentile
        pred_50 = pred_raw[:, 1]  # 50th percentile (median)
        pred_90 = pred_raw[:, 2]  # 90th percentile
    
    # Actual target values
    actual = target_seq[idx, :, 0]  # Shape: (future_seq_len,)
    
    # Plot predictions - MATCHING ORIGINAL REPOSITORY FORMAT
    # Original: ax2.plot(net_out[:,0], color = "red")  # 10th quantile
    #           ax2.plot(net_out[:,1], color = "blue")  # 50th quantile (median)
    #           ax2.plot(net_out[:,2], color = "red")   # 90th quantile
    #           ax2.plot(given_data[3].cpu().detach().numpy()[0], label = "target", color = "orange")
    ax.plot(future_x, pred_10, 'r-', linewidth=1.5, label='10% Quantile', zorder=5)  # Red for 10th
    ax.plot(future_x, pred_50, 'b-', linewidth=2, label='Prediction (Median)', zorder=5)  # Blue for median
    ax.plot(future_x, pred_90, 'r-', linewidth=1.5, label='90% Quantile', zorder=5)  # Red for 90th
    ax.plot(future_x, actual, 'orange', linewidth=2, label='Actual', zorder=6)  # Orange for target
    
    # Fill between quantiles
    ax.fill_between(future_x, pred_10, pred_90, alpha=0.2, color='red', zorder=3)
    
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlim(-past_seq_len - 5, future_seq_len + 5)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Normalized Close Price')
    ax.set_title(f'Test Case {idx + 1}: Network Output Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
results_dir = workspace_root / "results"
results_dir.mkdir(exist_ok=True)
plt.savefig(str(results_dir / 'test_predictions.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {results_dir / 'test_predictions.png'}")

# ============================================================================
# PLOT 3: Variable Selection Weights
# ============================================================================
print("\nGenerating variable selection weights plot...")

# Calculate average variable selection weights
# vs_weights shape: (batch_size, seq_len, n_variables, 1)
vs_weights_np = torch.mean(torch.mean(vs_weights, dim=0), dim=0).squeeze().cpu().numpy()

# Variable names
variable_names = ['Open', 'High', 'Low', 'Close', 'Hour']

fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
bars = ax3.bar(variable_names, vs_weights_np, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
ax3.set_title('Variable Selection Weights', fontsize=14, fontweight='bold')
ax3.set_ylabel('Weight')
ax3.set_xlabel('Variable')
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, vs_weights_np)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.3f}',
             ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=0)
plt.tight_layout()
results_dir = workspace_root / "results"
results_dir.mkdir(exist_ok=True)
plt.savefig(str(results_dir / 'variable_weights.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {results_dir / 'variable_weights.png'}")

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)

# Fix quantile order for all predictions (model outputs reversed)
pred_10_all = predictions[:, :, 2] if predictions[0, 0, 0] > predictions[0, 0, 1] else predictions[:, :, 0]
pred_50_all = predictions[:, :, 1]
pred_90_all = predictions[:, :, 0] if predictions[0, 0, 0] > predictions[0, 0, 1] else predictions[:, :, 2]
actual_all = target_seq[:, :, 0]

# Calculate prediction errors (using median)
mae = np.mean(np.abs(pred_50_all - actual_all))
rmse = np.sqrt(np.mean((pred_50_all - actual_all)**2))

# Calculate quantile coverage
within_band = (actual_all >= pred_10_all) & (actual_all <= pred_90_all)
coverage = np.mean(within_band) * 100
below_10 = np.mean(actual_all < pred_10_all) * 100
above_90 = np.mean(actual_all > pred_90_all) * 100

print(f"\nPrediction Metrics:")
print(f"  Mean Absolute Error (MAE): {mae:.6f}")
print(f"  Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"  Test Loss: {loss.item():.6f}")

print(f"\nQuantile Coverage Analysis:")
print(f"  Coverage (10%-90% band): {coverage:.2f}% (target: ~80%)")
print(f"  Below 10% quantile: {below_10:.2f}% (target: ~10%)")
print(f"  Above 90% quantile: {above_90:.2f}% (target: ~10%)")
if coverage < 70:
    print(f"  ⚠ Warning: Low coverage - actual values frequently outside prediction band")
elif coverage > 90:
    print(f"  ⚠ Note: Very high coverage - prediction bands may be too wide")
else:
    print(f"  ✓ Coverage is reasonable")

print(f"\nVariable Importance:")
total_weight = vs_weights_np.sum()
for name, weight in zip(variable_names, vs_weights_np):
    print(f"  {name:8s}: {weight:.4f} ({100*weight/total_weight:.2f}%)")

print(f"\nGenerated Files:")
print(f"  - training_losses.png")
print(f"  - test_predictions.png")
print(f"  - variable_weights.png")
print("="*60)

plt.show()

