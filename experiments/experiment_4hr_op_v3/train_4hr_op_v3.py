"""
Training script for experiment_4hr_op_v3

This version predicts 4-hour returns (percent moves) instead of raw Close.
It applies per-column normalisation (train split only) and keeps the
short-horizon forecast (8 hours ahead).
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent
workspace_root = script_dir.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import torch
from Network import TFN
from data import load_data_from_csv, get_batches
from utils import forward_pass
from scalers import compute_stats, save_stats, validate_stats, round_trip_test, load_stats
import pandas as pd
import matplotlib.pyplot as plt
import signal
import atexit
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
CSV_PATH = str(workspace_root / "data/btcusd_4hr_resampled.csv")

# Continuous features (same 14 as 5min_op_v3)
continuous_columns = [
    "Open", "High", "Low", "Close", "Volume",
    "returns", "atr_10", "volatility_20",
    "rsi_14", "bollinger_upper", "bollinger_lower", "bollinger_position", "bollinger_width",
    "momentum_5",
]

discrete_columns = ["Hour"]
target_columns = ["returns"]  # predict returns

n_variables_past_cont = len(continuous_columns)
n_variables_future_cont = 0
n_variables_past_disc = [24]
n_variables_future_disc = [24]

# Hyperparameters (mirrors 5min_op_v3)
batch_size = 160
test_batch_size = 160
dim_model = 160
n_lstm_layers = 4
n_attention_layers = 3
n_heads = 6
learning_rate = 0.0003
dropout_rate = 0.3

# Sequence lengths (4-hour candle)
past_seq_len = 80   # 320 hours (~13.3 days)
future_seq_len = 2 # 8 hours ahead

# Device selection
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

quantiles = torch.tensor([0.1, 0.5, 0.9]).float().to(device)

print("=" * 70)
print("EXPERIMENT 4HR_OP_V3 - Training Configuration")
print("=" * 70)
print(f"Using device: {device}")
print(f"Features: {len(continuous_columns)} continuous + {len(discrete_columns)} discrete")
print(f"Sequence lengths: past={past_seq_len}, future={future_seq_len}")
print(f"Hyperparameters: lr={learning_rate}, dropout={dropout_rate}")
print("=" * 70)

# ============================================================================
# DATA LOADING & FEATURE ENGINEERING
# ============================================================================
sys.path.insert(0, str(script_dir))
from feature_engineering import add_all_features  # noqa: E402

print("\nLoading data...")
full_data = load_data_from_csv(CSV_PATH)
print("Calculating technical indicators...")
full_data = add_all_features(full_data)

# Drop rows with NaNs in features (returns introduces NaN at first row)
full_data = full_data.dropna(subset=continuous_columns)
full_data = full_data.reset_index(drop=True)

# Train / Test split
train = full_data[(full_data["timestamp"] >= "2017-01-01") & (full_data["timestamp"] < "2024-01-01")].copy()
test = full_data[(full_data["timestamp"] >= "2024-01-01") & (full_data["timestamp"] < "2025-01-01")].copy()

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

print(f"Training rows: {len(train)} ({train['timestamp'].min()} → {train['timestamp'].max()})")
print(f"Test rows: {len(test)} ({test['timestamp'].min()} → {test['timestamp'].max()})")

# ============================================================================
# NORMALISATION (train split only)
# ============================================================================
norm_path = script_dir / "norm_stats.json"
if norm_path.exists():
    print(f"\nLoading existing normalization stats from {norm_path}")
    norm_stats = load_stats(str(norm_path))
    validate_stats(norm_stats)
else:
    print("\nComputing per-column normalization stats (train split)...")
    norm_stats = compute_stats(train, continuous_columns)
    validate_stats(norm_stats)
    save_stats(norm_stats, str(norm_path))
    print("Stats saved.")
    print("\nRound-trip test on Close column...")
    round_trip_test(train, "Close", norm_stats)

print(f"\nNormalization summary: Close mean={norm_stats['Close'].mean:.2f}, std={norm_stats['Close'].std:.2f}")
print(f"returns mean={norm_stats['returns'].mean:.6f}, std={norm_stats['returns'].std:.6f}")

# ============================================================================
# MODEL INITIALISATION
# ============================================================================
print("\nInitialising model...")
model = TFN(
    n_variables_past_cont,
    n_variables_future_cont,
    n_variables_past_disc,
    n_variables_future_disc,
    dim_model,
    n_quantiles=quantiles.shape[0],
    dropout_r=dropout_rate,
    n_attention_layers=n_attention_layers,
    n_lstm_layers=n_lstm_layers,
    n_heads=n_heads,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

models_dir = script_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)
losses, test_losses = [], []

# Load specific checkpoint
checkpoint_path = models_dir / "model_146400.pt"
if checkpoint_path.exists():
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        losses = checkpoint.get("losses", [])
        test_losses = checkpoint.get("test_losses", [])
        print(f"Resuming from checkpoint: {checkpoint_path}")
        print(f"Loaded {len(losses)} training steps, {len(test_losses)} test evaluations")
    except Exception as exc:
        print(f"Could not load checkpoint {checkpoint_path}: {exc}\nStarting fresh.")
else:
    print(f"Checkpoint {checkpoint_path} not found. Looking for latest checkpoint...")
    checkpoint_files = sorted(models_dir.glob("model_*.pt"))
    if checkpoint_files:
        latest = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        try:
            checkpoint = torch.load(latest, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            losses = checkpoint.get("losses", [])
            test_losses = checkpoint.get("test_losses", [])
            print(f"Resuming from latest checkpoint: {latest}")
        except Exception as exc:
            print(f"Could not load checkpoint {latest}: {exc}\nStarting fresh.")
    else:
        print("No checkpoints found, starting fresh.")

# ============================================================================
# DATA GENERATORS (per-column normalised)
# ============================================================================
train_gen = get_batches(
    train,
    in_seq_len=past_seq_len,
    out_seq_len=future_seq_len,
    con_cols=continuous_columns,
    disc_cols=discrete_columns,
    target_cols=target_columns,
    batch_size=batch_size,
    normalise=True,
    stats=norm_stats,
    device=device,
)

test_gen = get_batches(
    test,
    in_seq_len=past_seq_len,
    out_seq_len=future_seq_len,
    con_cols=continuous_columns,
    disc_cols=discrete_columns,
    target_cols=target_columns,
    batch_size=test_batch_size,
    normalise=True,
    stats=norm_stats,
    device=device,
)

# ============================================================================
# TRAINING LOOP
# ============================================================================
total_steps = 200_000
test_eval_frequency = 50
save_frequency = 400

start_step = len(losses)
print("\nStarting training...")
print(f"Starting at step {start_step}")

if start_step >= total_steps:
    print("Training already complete.")
    raise SystemExit

def emergency_save():
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
        "test_losses": test_losses,
        "step": len(losses),
    }
    path = models_dir / f"model_emergency_{len(losses)}.pt"
    torch.save(checkpoint, path)
    print(f"Emergency checkpoint saved to {path}")

atexit.register(emergency_save)

def handle_signal(signum, frame):
    print("\nSignal received. Saving and exiting...")
    emergency_save()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

model.train()

try:
    for step in range(start_step, total_steps):
        loss, _, _, _ = forward_pass(
            model, train_gen, batch_size, quantiles,
            discrete_dims=n_variables_past_disc,
            device=device,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

        if step % test_eval_frequency == 0:
            model.eval()
            with torch.no_grad():
                test_loss, _, _, _ = forward_pass(
                    model, test_gen, test_batch_size, quantiles,
                    discrete_dims=n_variables_past_disc,
                    device=device,
                )
                test_losses.append(float(test_loss.detach().cpu()))
            model.train()
            print(f"Step {step}: Train loss={losses[-1]:.6f}, Test loss={test_losses[-1]:.6f}")

        if (step + 1) % save_frequency == 0 or (step + 1) == total_steps:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
                "test_losses": test_losses,
                "step": step + 1,
            }
            path = models_dir / f"model_{step + 1}.pt"
            torch.save(checkpoint, path)
            print(f"Saved checkpoint: {path}")

except KeyboardInterrupt:
    print("Training interrupted by user.")
    emergency_save()
    raise

except Exception as exc:
    print(f"Unexpected error: {exc}")
    emergency_save()
    raise

# ============================================================================
# FINAL SAVE & PLOTS
# ============================================================================
final_checkpoint = models_dir / f"model_{len(losses)}.pt"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
        "test_losses": test_losses,
        "step": len(losses),
    },
    final_checkpoint,
)
print(f"\nTraining complete. Final checkpoint saved to {final_checkpoint}")

results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(losses[250:], label="Training Loss", linewidth=1)
ax1.set_title("Training Loss", fontsize=14, fontweight="bold")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)
ax1.legend()

if len(test_losses) > 5:
    ax2.plot(test_losses[5:], label="Test Loss", linewidth=1, color="orange")
    ax2.set_title("Test Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Evaluation Step")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

plt.tight_layout()
loss_plot = results_dir / "training_losses_4hr_op_v3.png"
plt.savefig(loss_plot, dpi=150, bbox_inches="tight")
print(f"Loss curves saved to {loss_plot}")

print("=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(f"Steps completed: {len(losses)} / {total_steps}")
if losses:
    print(f"Final train loss: {losses[-1]:.6f}")
if test_losses:
    print(f"Final test loss: {test_losses[-1]:.6f}")
print(f"Completed at: {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 70)
