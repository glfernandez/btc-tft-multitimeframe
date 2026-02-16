# Quick Start

## 1. Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Add Required Artifacts
For each timeframe (`15min`, `4hr`, `1day`, `1week`), ensure:
- `experiments/experiment_<tf>_op_v3/models/model_200000.pt`
- `experiments/experiment_<tf>_op_v3/norm_stats.json`

Provide a 1-minute BTC/USDT CSV and pass it via `--raw-data-path`.

## 3. Run Inference
Individual timeframe:
```bash
python scripts/infer_cli.py \
  --timestamp "2024-11-20 12:00:00" \
  --mode individual \
  --timeframe 15min \
  --raw-data-path /absolute/path/to/btc_1min.csv
```

Ensemble:
```bash
python scripts/infer_cli.py \
  --timestamp "2024-11-20 12:00:00" \
  --mode ensemble \
  --include-futureview \
  --include-mpc \
  --raw-data-path /absolute/path/to/btc_1min.csv \
  --output outputs/ensemble.json
```

## 4. Train Models
One timeframe:
```bash
python scripts/train_timeframe.py --timeframe 4hr
```

All timeframes:
```bash
python scripts/train_timeframe.py --all
```

## 5. Public Publishing Rule
Before pushing public updates:
- do not commit raw datasets
- do not commit API keys/tokens
- do not commit local absolute paths in logs or metadata
