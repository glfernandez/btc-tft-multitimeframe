# Public Usage Guide

## Inference Modes

### Individual timeframe
Use when you want one horizon only (`15min`, `4hr`, `1day`, `1week`):
```bash
python scripts/infer_cli.py \
  --timestamp '2024-11-20 12:00:00' \
  --mode individual \
  --timeframe 15min
```

### Ensemble inference
Use when you want all horizons together:
```bash
python scripts/infer_cli.py \
  --timestamp '2024-11-20 12:00:00' \
  --mode ensemble
```

### Ensemble + adapters
FutureView-style fields and MPC vector:
```bash
python scripts/infer_cli.py \
  --timestamp '2024-11-20 12:00:00' \
  --mode ensemble \
  --include-futureview \
  --include-mpc \
  --output ensemble_output.json
```

## Train Further

### One timeframe
```bash
python scripts/train_timeframe.py --timeframe 4hr
```

### All timeframes
```bash
python scripts/train_timeframe.py --all
```

## Build a Clean Public Release Folder
Create a sanitized copy for publishing:
```bash
python scripts/create_public_release.py --output ../btc_tft_public_release
```

Include checkpoints if you want immediate inference without retraining:
```bash
python scripts/create_public_release.py \
  --output ../btc_tft_public_release \
  --include-checkpoints
```

## Troubleshooting
- If inference appears stalled on cloud drives, copy repo + data to local SSD path and run again.
- Ensure all `model_200000.pt` and `norm_stats.json` files exist in each experiment folder.
- Ensure the 1-minute BTC/USDT CSV exists under `data/` or pass `--raw-data-path`.
