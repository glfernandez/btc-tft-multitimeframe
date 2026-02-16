# BTC/USDT Multi-Timeframe TFT

Temporal Fusion Transformer (TFT) forecasting pipeline for BTC/USDT with:
- individual timeframe inference (`15min`, `4hr`, `1day`, `1week`)
- ensemble inference across all timeframes
- adapters for MPC/FutureView integration

## Project Layout
- `scripts/inference_pipeline.py`: core multi-timeframe predictor
- `scripts/futureview_adapter.py`: converts ensemble output to FutureView-style fields
- `scripts/mpc_adapter.py`: converts ensemble output to MPC-ready vectors/paths
- `scripts/infer_cli.py`: public CLI for individual or ensemble inference
- `scripts/train_timeframe.py`: public CLI wrapper for further training
- `experiments/experiment_*_op_v3/`: timeframe-specific configs, features, and model artifacts

## Quickstart

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify required artifacts exist
You need these per timeframe:
- `experiments/experiment_<tf>_op_v3/feature_engineering.py`
- `experiments/experiment_<tf>_op_v3/norm_stats.json`
- `experiments/experiment_<tf>_op_v3/models/model_200000.pt`

And a raw 1-minute BTC/USDT CSV at:
- `data/btcusd_2012-01-01_to_2024-11-23_1min_updated_20250528.csv`

### 3. Inference (individual timeframe)
```bash
python scripts/infer_cli.py \
  --timestamp '2024-11-20 12:00:00' \
  --mode individual \
  --timeframe 4hr
```

### 4. Inference (ensemble)
```bash
python scripts/infer_cli.py \
  --timestamp '2024-11-20 12:00:00' \
  --mode ensemble \
  --include-futureview \
  --include-mpc
```

### 5. Save inference JSON
```bash
python scripts/infer_cli.py \
  --timestamp '2024-11-20 12:00:00' \
  --mode ensemble \
  --output inference_output.json
```

## Train Further
Use timeframe wrapper:
```bash
python scripts/train_timeframe.py --timeframe 15min
python scripts/train_timeframe.py --timeframe 4hr
python scripts/train_timeframe.py --timeframe 1day
python scripts/train_timeframe.py --timeframe 1week
```

Or run all sequentially:
```bash
python scripts/train_timeframe.py --all
```

## Notes
- This repository intentionally separates reusable source from heavy local outputs.
- For public publishing, avoid committing raw data, logs, and checkpoints unless intentionally sharing them.
