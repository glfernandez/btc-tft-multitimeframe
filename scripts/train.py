"""
Training script for Temporal Fusion Transformer adapted for custom CSV data format.

This script trains a TFT model on your BTC/USD 1-minute data.
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
from utils import forward_pass, QuantileLoss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import traceback
import signal
import atexit
from datetime import datetime

# Configuration
CSV_PATH = str(workspace_root / "data/btcusd_2012-01-01_to_2024-11-23_1min_updated_20250528.csv")

# Define which columns are used as continuous and discrete input, as well as prediction targets
continuous_columns = ['Open', 'High', 'Low', 'Close']
discrete_columns = ['Hour']  # Can add 'Day', 'Month' if needed
target_columns = ['Close']

# Input data shape
n_variables_past_continuous = len(continuous_columns)
n_variables_future_continuous = 0
n_variables_past_discrete = [24]  # 24 hours. Use [24, 31, 12] for Hour, Day, Month
n_variables_future_discrete = [24]

# Hyperparameters
batch_size = 160
test_batch_size = 160
dim_model = 160
n_lstm_layers = 4
n_attention_layers = 3
n_heads = 6
learning_rate = 0.0005
dropout_rate = 0.2

# Sequence lengths
past_seq_len = 80  # Look back 80 minutes
future_seq_len = 15  # Predict 15 minutes ahead

# Quantiles for quantile loss
# Device selection: MPS (Apple Silicon) > CUDA > CPU
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

quantiles = torch.tensor([0.1, 0.5, 0.9]).float().to(device)

print(f"Using device: {device}")

# Load data
print("Loading data...")
full_data = load_data_from_csv(CSV_PATH)

# Split data into train and test sets
# Original repository used: Training = 2018-2019, Testing = 2020
# This matches the original setup for comparison with their figures/plots
train_start_date = '2018-01-01'
train_end_date = '2020-01-01'  # Exclusive - so includes 2019-12-31
test_start_date = '2020-01-01'
test_end_date = '2021-01-01'  # Testing on 2020 data

train_data = full_data[(full_data['timestamp'] >= train_start_date) & 
                        (full_data['timestamp'] < train_end_date)].copy()
test_data = full_data[(full_data['timestamp'] >= test_start_date) & 
                       (full_data['timestamp'] < test_end_date)].copy()

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
if len(train_data) > 0:
    print(f"Training date range: {train_data['timestamp'].min()} to {train_data['timestamp'].max()}")
if len(test_data) > 0:
    print(f"Test date range: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")
if len(train_data) == 0 or len(test_data) == 0:
    print("WARNING: One or both datasets are empty. Check date ranges match your data.")

# Initialize model
print("Initializing model...")
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
)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Check if we should load from checkpoint
import glob
import os
# Look for checkpoints in models/ directory
models_dir = workspace_root / "models"
checkpoint_files = list(models_dir.glob("model_*.pt")) if models_dir.exists() else []
# Also check root directory for backward compatibility
checkpoint_files.extend(glob.glob(str(workspace_root / "model_*.pt")))
if checkpoint_files:
    # Get the latest checkpoint
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    load_model = True
    checkpoint_path = str(latest_checkpoint) if isinstance(latest_checkpoint, Path) else latest_checkpoint
    print(f"Found existing checkpoint: {checkpoint_path}")
else:
    load_model = False
    checkpoint_path = None
    print("No existing checkpoints found, starting from scratch")

losses = []
test_losses = []

if load_model:
    try:
        # Handle PyTorch 2.6+ weights_only requirement
        import torch._utils
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            # Older PyTorch versions don't have weights_only parameter
            checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle both old format (model_state) and new format (model_state_dict)
        if 'model_state' in checkpoint:
            model = checkpoint['model_state']
            model = model.to(device)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise KeyError("Checkpoint missing model state")
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        losses = checkpoint.get('losses', [])
        test_losses = checkpoint.get('test_losses', [])
        print(f"Loaded model from checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        print("Starting training from scratch...")
else:
    print("Starting training from scratch...")

# Create data generators
train_gen = get_batches(
    train_data,
    in_seq_len=past_seq_len,
    out_seq_len=future_seq_len,
    con_cols=continuous_columns,
    disc_cols=discrete_columns,
    target_cols=target_columns,
    batch_size=batch_size,
    device=device
)

test_gen = get_batches(
    test_data,
    in_seq_len=past_seq_len,
    out_seq_len=future_seq_len,
    con_cols=continuous_columns,
    disc_cols=discrete_columns,
    target_cols=target_columns,
    batch_size=test_batch_size,
    norm=train_data,  # Use training data statistics for normalization
    device=device
)

# Training loop
print("Starting training...")
total_steps = 200000  # Target: 200,000 steps total (like original repo's model_100000)
test_eval_frequency = 50
save_frequency = 400

# Calculate starting step (from checkpoint if loaded)
start_step = len(losses) if losses else 0
remaining_steps = total_steps - start_step

if start_step > 0:
    print(f"Resuming from step {start_step:,} / {total_steps:,}")
    print(f"Remaining steps: {remaining_steps:,}")
else:
    print(f"Starting fresh - target: {total_steps:,} steps")

if remaining_steps <= 0:
    print(f"Training already completed! ({start_step:,} steps completed, target: {total_steps:,})")
    print("If you want to continue training, increase total_steps or start from scratch.")
    exit(0)

# Emergency save function
def emergency_save():
    """Save checkpoint in case of unexpected termination"""
    try:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'losses': losses,
            'test_losses': test_losses
        }
        emergency_path = str(models_dir / f"model_emergency_{len(losses)}.pt")
        torch.save(checkpoint, emergency_path)
        print(f"\nEmergency checkpoint saved: {emergency_path}")
    except Exception as e:
        print(f"Failed to save emergency checkpoint: {e}")

# Register emergency save handlers
atexit.register(emergency_save)

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully"""
    print("\n\nInterrupt received! Saving checkpoint...")
    emergency_save()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Training loop with error handling
model.train()
last_successful_step = start_step
consecutive_errors = 0
max_consecutive_errors = 10

try:
    for step in range(start_step, total_steps):
        try:
            # Run model against test set periodically
            if step % test_eval_frequency == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, _, _, _ = forward_pass(
                        model, test_gen, test_batch_size, quantiles,
                        discrete_dims=n_variables_past_discrete,
                        device=device
                    )
                    test_losses.append(test_loss.cpu().detach().numpy())
                model.train()
                print(f"Step {step}: Train loss = {losses[-1] if losses else 'N/A'}, Test loss = {test_losses[-1] if test_losses else 'N/A'}")

            # Save model periodically
            if step % save_frequency == 0 and step > 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'losses': losses,
                    'test_losses': test_losses
                }
                save_path = str(models_dir / f"model_{len(losses)}.pt")
                torch.save(checkpoint, save_path)
                print(f"Saved checkpoint: {save_path}")
                last_successful_step = step
                consecutive_errors = 0  # Reset error counter on successful save

            # Forward pass
            optimizer.zero_grad()
            loss, net_out, vs_weights, given_data = forward_pass(
                model, train_gen, batch_size, quantiles,
                discrete_dims=n_variables_past_discrete,
                device=device
            )

            # Backward pass
            loss_value = loss.cpu().detach().numpy()
            losses.append(loss_value)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Clean up
            del loss, net_out, vs_weights, given_data
            
            # Periodic memory cleanup for MPS
            if device == 'mps' and step % 1000 == 0:
                torch.mps.empty_cache()
                
            consecutive_errors = 0  # Reset on success
            
        except RuntimeError as e:
            consecutive_errors += 1
            error_msg = str(e)
            print(f"\n⚠ ERROR at step {step}: {error_msg}")
            print(f"Traceback:\n{traceback.format_exc()}")
            
            # Check for memory errors
            if "out of memory" in error_msg.lower() or "mps" in error_msg.lower():
                print("\n⚠ Memory error detected! Attempting recovery...")
                torch.mps.empty_cache()
                # Reduce batch size temporarily if needed
                print("Cleared MPS cache. Continuing...")
            else:
                print(f"\n⚠ Runtime error. Attempting to continue...")
            
            # Save checkpoint on error
            if consecutive_errors <= 3:  # Save checkpoint if not too many consecutive errors
                try:
                    emergency_save()
                except:
                    pass
            
            # Stop if too many consecutive errors
            if consecutive_errors >= max_consecutive_errors:
                print(f"\n❌ Too many consecutive errors ({consecutive_errors}). Stopping training.")
                print(f"Last successful step: {last_successful_step}")
                emergency_save()
                break
                
        except Exception as e:
            consecutive_errors += 1
            print(f"\n❌ UNEXPECTED ERROR at step {step}: {type(e).__name__}: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")
            
            # Save emergency checkpoint
            try:
                emergency_save()
            except:
                pass
            
            if consecutive_errors >= max_consecutive_errors:
                print(f"\n❌ Too many consecutive errors ({consecutive_errors}). Stopping training.")
                break

except KeyboardInterrupt:
    print("\n\n⚠ Training interrupted by user!")
    emergency_save()
except Exception as e:
    print(f"\n\n❌ Fatal error: {type(e).__name__}: {e}")
    print(f"Traceback:\n{traceback.format_exc()}")
    emergency_save()
    raise

# Final save
if len(losses) > last_successful_step:
    try:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'losses': losses,
            'test_losses': test_losses
        }
        final_path = str(models_dir / f"model_{len(losses)}.pt")
        torch.save(checkpoint, final_path)
        print(f"\nFinal checkpoint saved: {final_path}")
    except Exception as e:
        print(f"Warning: Could not save final checkpoint: {e}")

print("\n" + "="*60)
if len(losses) >= total_steps:
    print("✅ Training completed successfully!")
else:
    print(f"⚠ Training stopped at step {len(losses)}")
print(f"Total steps completed: {len(losses):,} / {total_steps:,} ({100*len(losses)/total_steps:.1f}%)")
if losses:
    print(f"Final training loss: {losses[-1]}")
if test_losses:
    print(f"Final test loss: {test_losses[-1]}")
print(f"Latest checkpoint: model_{len(losses)}.pt")
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

