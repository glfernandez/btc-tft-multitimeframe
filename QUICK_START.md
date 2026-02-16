# Quick Start Guide - Organized Workspace

## 🎯 Current Structure

```
btc_tft/
├── scripts/          # 13 Python scripts
├── models/           # 1 checkpoint (model_200000.pt)
├── results/          # 8 plots/results files
├── docs/             # 16 documentation files
├── experiments/      # Template for new experiments
└── data/            # Data files
```

## 🚀 Common Tasks

### Run Training
```bash
python scripts/train.py
```
- Automatically finds checkpoints in `models/`
- Saves new checkpoints to `models/`
- Uses data from `data/`

### Evaluate Model
```bash
python scripts/evaluate.py
```
- Loads model from `models/model_200000.pt`
- Saves plots to `results/`

### Make Predictions
```bash
python scripts/predict.py
```
- Uses model from `models/`
- Saves predictions to `results/`

## 📦 Starting a New Experiment

### Example: Train with 400/75 Sequence Lengths

1. **Create experiment folder**:
   ```bash
   cp -r experiments/EXPERIMENT_TEMPLATE experiments/experiment_400_75
   ```

2. **Copy and modify training script**:
   ```bash
   cp scripts/train.py experiments/experiment_400_75/train_400_75.py
   ```

3. **Edit `train_400_75.py`**:
   ```python
   # Change sequence lengths
   past_seq_len = 400  # Instead of 80
   future_seq_len = 75  # Instead of 15
   
   # Update models directory
   models_dir = workspace_root / "experiments" / "experiment_400_75" / "models"
   models_dir.mkdir(parents=True, exist_ok=True)
   ```

4. **Run**:
   ```bash
   cd experiments/experiment_400_75
   python train_400_75.py
   ```

5. **Results saved** in `experiments/experiment_400_75/`

## 📁 Directory Purposes

| Directory | Purpose | Contents |
|-----------|---------|----------|
| `scripts/` | All Python scripts | train.py, evaluate.py, predict.py, etc. |
| `models/` | Model checkpoints | model_*.pt files |
| `results/` | Evaluation results | Plots (*.png), predictions (*.csv) |
| `docs/` | Documentation | README, guides, analysis docs |
| `experiments/` | Training variations | Individual experiment folders |
| `data/` | Data files | CSV files, datasets |

## ✅ Benefits

- ✅ **Clean workspace** - No clutter in root directory
- ✅ **Easy experiments** - Each variation is self-contained
- ✅ **Organized results** - All outputs in dedicated folders
- ✅ **Reusable scripts** - Scripts work from any location
- ✅ **Scalable** - Easy to add new experiments

## 🔧 Script Paths

All scripts automatically:
- Add workspace root to Python path
- Find data in `data/` directory
- Save checkpoints to `models/` directory
- Save results to `results/` directory

You can run scripts from anywhere:
```bash
# From workspace root
python scripts/train.py

# From scripts directory
cd scripts
python train.py

# From experiment folder
cd experiments/experiment_400_75
python train_400_75.py
```

All paths are handled automatically!

