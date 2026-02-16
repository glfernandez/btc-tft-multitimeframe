"""
Multi-Timeframe Inference Pipeline

This script provides a unified interface to run predictions from all three models
(15min, 4hr, 1day) on raw 1-minute data.

Usage:
    from scripts.inference_pipeline import MultiTimeframePredictor
    
    predictor = MultiTimeframePredictor()
    predictions = predictor.predict_at_timestamp('2024-11-20 12:00:00')
    
    # Returns:
    # {
    #     '15min': {'lower': [...], 'median': [...], 'upper': [...], 'last_close': ...},
    #     '4hr': {...},
    #     '1day': {...}
    # }
"""

import sys
from pathlib import Path

workspace_root = Path(__file__).parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import pandas as pd
import numpy as np
import torch
import json
from datetime import datetime, timedelta

from Network import TFN
from data import load_data_from_csv, one_hot
from scalers import load_stats

# Import feature engineering from each experiment
sys.path.insert(0, str(workspace_root / "experiments/experiment_15min_op_v3"))
from feature_engineering import add_all_features as add_features_15min  # noqa: E402

sys.path.insert(0, str(workspace_root / "experiments/experiment_4hr_op_v3"))
from feature_engineering import add_all_features as add_features_4hr  # noqa: E402

sys.path.insert(0, str(workspace_root / "experiments/experiment_1day_op_v3"))
from feature_engineering import add_all_features as add_features_1day  # noqa: E402

sys.path.insert(0, str(workspace_root / "experiments/experiment_1week_op_v3"))
from feature_engineering import add_all_features as add_features_1week  # noqa: E402


class MultiTimeframePredictor:
    """
    Multi-timeframe predictor that runs inference on 15min, 4hr, 1day, and 1week models.
    """
    
    def __init__(self, 
                 raw_data_path=None,
                 device=None):
        """
        Initialize the multi-timeframe predictor.
        
        Args:
            raw_data_path: Path to 1-minute data CSV. If None, uses default.
            device: Device to run on ('mps', 'cuda', 'cpu'). Auto-detects if None.
        """
        self.workspace_root = workspace_root
        
        # Device selection
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Default raw data path
        if raw_data_path is None:
            raw_data_path = workspace_root / "data/btcusd_2012-01-01_to_2024-11-23_1min_updated_20250528.csv"
        self.raw_data_path = raw_data_path
        
        # Model configurations
        self.configs = {
            '15min': {
                'experiment_dir': workspace_root / "experiments/experiment_15min_op_v3",
                'discrete_columns': ['Hour'],
                'discrete_dims': [24],
                'resample_freq': '15min',
                'add_features': add_features_15min,
            },
            '4hr': {
                'experiment_dir': workspace_root / "experiments/experiment_4hr_op_v3",
                'discrete_columns': ['Hour'],
                'discrete_dims': [24],
                'resample_freq': '4h',
                'add_features': add_features_4hr,
            },
            '1day': {
                'experiment_dir': workspace_root / "experiments/experiment_1day_op_v3",
                'discrete_columns': ['DayOfWeek', 'DayOfMonth', 'Month'],
                'discrete_dims': [7, 31, 12],
                'resample_freq': '1D',
                'add_features': add_features_1day,
            },
            '1week': {
                'experiment_dir': workspace_root / "experiments/experiment_1week_op_v3",
                'discrete_columns': ['DayOfWeek', 'DayOfMonth', 'Month'],
                'discrete_dims': [7, 31, 12],
                'resample_freq': '1W',
                'add_features': add_features_1week,
                'past_seq_len': 40,  # Reduced for weekly data
            }
        }
        
        # Continuous features (same for all models)
        self.continuous_columns = [
            "Open", "High", "Low", "Close", "Volume",
            "returns", "atr_10", "volatility_20",
            "rsi_14", "bollinger_upper", "bollinger_lower", "bollinger_position", "bollinger_width",
            "momentum_5",
        ]
        
        self.target_columns = ["returns"]
        # past_seq_len varies by timeframe (80 for 15min/4hr/1day, 40 for 1week)
        # Will use config-specific values when needed
        self.future_seq_len = 2
        self.quantiles = torch.tensor([0.1, 0.5, 0.9]).float().to(self.device)
        
        # Expected bar durations in seconds for each timeframe
        self.expected_bar_seconds = {
            '15min': 15 * 60,      # 900 seconds
            '4hr': 4 * 60 * 60,    # 14400 seconds
            '1day': 24 * 60 * 60,  # 86400 seconds
            '1week': 7 * 24 * 60 * 60  # 604800 seconds
        }
        
        # Load models and stats
        print("=" * 70)
        print("INITIALIZING MULTI-TIMEFRAME PREDICTOR")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Raw data: {self.raw_data_path}")
        print()
        
        self.models = {}
        self.norm_stats = {}
        self.calibration_params = {}
        
        for timeframe in ['15min', '4hr', '1day', '1week']:
            print(f"Loading {timeframe} model...")
            self._load_model(timeframe)
            print(f"  ✓ {timeframe} model loaded")
        
        print("\n✓ All models loaded successfully")
        print("=" * 70)
        
        # Cache for resampled data
        self._raw_data = None
        self._resampled_data = {}
    
    def _load_model(self, timeframe):
        """Load model, normalization stats, and calibration parameters for a timeframe."""
        config = self.configs[timeframe]
        exp_dir = config['experiment_dir']
        
        # Load model
        model = TFN(
            len(self.continuous_columns),
            0,
            config['discrete_dims'],
            config['discrete_dims'],
            160,
            n_quantiles=3,
            dropout_r=0.3,
            n_attention_layers=3,
            n_lstm_layers=4,
            n_heads=6,
        ).to(self.device)
        
        # Load checkpoint
        checkpoint_path = exp_dir / "models/model_200000.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.models[timeframe] = model
        
        # Load normalization stats
        norm_path = exp_dir / "norm_stats.json"
        self.norm_stats[timeframe] = load_stats(str(norm_path))
        
        # Load calibration parameters (dynamic calibration)
        calib_path = exp_dir / "calibration_offsets_dynamic.json"
        if calib_path.exists():
            with open(calib_path) as f:
                self.calibration_params[timeframe] = json.load(f)
        else:
            self.calibration_params[timeframe] = None
    
    def _load_raw_data(self):
        """Load and cache raw 1-minute data."""
        if self._raw_data is None:
            print("Loading raw 1-minute data...")
            self._raw_data = load_data_from_csv(str(self.raw_data_path))
            print(f"  Loaded {len(self._raw_data):,} rows")
        return self._raw_data
    
    def _resample_data(self, timeframe, end_timestamp):
        """
        Resample raw data to the specified timeframe, up to end_timestamp.
        
        Args:
            timeframe: '15min', '4hr', '1day', or '1week'
            end_timestamp: pd.Timestamp - end time for resampling
        
        Returns:
            Resampled DataFrame with features
        """
        if timeframe in self._resampled_data:
            # Check if cached data covers end_timestamp
            cached = self._resampled_data[timeframe]
            if len(cached) > 0 and cached['timestamp'].max() >= end_timestamp:
                # Return subset up to end_timestamp
                return cached[cached['timestamp'] <= end_timestamp].copy()
        
        # Load raw data
        raw_data = self._load_raw_data()
        
        # Filter to data up to end_timestamp
        raw_data = raw_data[raw_data['timestamp'] <= end_timestamp].copy()
        
        # Resample
        config = self.configs[timeframe]
        freq = config['resample_freq']
        
        raw_indexed = raw_data.set_index('timestamp')
        resampled = raw_indexed.resample(freq).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        resampled = resampled.reset_index()
        
        # Recreate basic time features
        resampled['Hour'] = resampled['timestamp'].apply(lambda x: x.hour)
        resampled['Day'] = resampled['timestamp'].apply(lambda x: x.day - 1)
        resampled['Month'] = resampled['timestamp'].apply(lambda x: x.month - 1)
        
        # Add features (this will add DayOfWeek, DayOfMonth, Month for 1day)
        # Make sure timestamp is datetime for feature engineering
        if not pd.api.types.is_datetime64_any_dtype(resampled['timestamp']):
            resampled['timestamp'] = pd.to_datetime(resampled['timestamp'])
        resampled = config['add_features'](resampled)
        
        # Fallback: Manually add discrete features for 1day and 1week if missing (in case feature engineering didn't add them)
        if timeframe in ['1day', '1week']:
            if 'DayOfWeek' not in resampled.columns:
                resampled['DayOfWeek'] = pd.to_datetime(resampled['timestamp']).dt.dayofweek
            if 'DayOfMonth' not in resampled.columns:
                resampled['DayOfMonth'] = pd.to_datetime(resampled['timestamp']).dt.day - 1
            # Month should already exist, but ensure it's correct
            if 'Month' not in resampled.columns or resampled['Month'].dtype != 'int64':
                resampled['Month'] = pd.to_datetime(resampled['timestamp']).dt.month - 1
        
        # Drop NaN rows (from feature calculation)
        resampled = resampled.dropna(subset=self.continuous_columns).reset_index(drop=True)
        
        # Cache
        self._resampled_data[timeframe] = resampled
        
        return resampled
    
    def _normalize_data(self, data, timeframe):
        """Normalize data using timeframe-specific stats."""
        from scalers import apply_z
        data = data.copy()
        apply_z(data, self.norm_stats[timeframe], cols=self.continuous_columns)
        return data
    
    def _prepare_input(self, data, timeframe):
        """
        Prepare input tensors for model inference.
        
        Args:
            data: DataFrame with last past_seq_len rows
            timeframe: '15min', '4hr', '1day', or '1week'
        
        Returns:
            Tuple of (in_seq_continuous, in_seq_discrete, future_discrete)
        """
        config = self.configs[timeframe]
        
        # Normalize
        data_norm = self._normalize_data(data, timeframe)
        
        # Get last past_seq_len rows (timeframe-specific)
        past_seq_len = config.get('past_seq_len', 80)  # Default 80, override for 1week
        if len(data_norm) > past_seq_len:
            data_window = data_norm.iloc[-past_seq_len:].copy()
        else:
            data_window = data_norm.copy()
        
        # Continuous features
        con_cols_idx = [data_window.columns.get_loc(col) for col in self.continuous_columns]
        in_seq_continuous = data_window.iloc[:, con_cols_idx].values
        
        # Discrete features
        # Check if columns exist (1day needs DayOfWeek, DayOfMonth, Month which are added by feature engineering)
        missing_cols = [col for col in config['discrete_columns'] if col not in data_window.columns]
        if missing_cols:
            raise ValueError(f"Missing discrete columns for {timeframe}: {missing_cols}. Available columns: {list(data_window.columns)}")
        
        disc_cols_idx = [data_window.columns.get_loc(col) for col in config['discrete_columns']]
        in_seq_discrete = data_window.iloc[:, disc_cols_idx].values
        
        # Future discrete features (use last values for now, or predict)
        # For inference, we'll use the last discrete values
        future_discrete = data_window.iloc[-self.future_seq_len:, disc_cols_idx].values
        if len(future_discrete) < self.future_seq_len:
            # Pad with last value
            last_row = future_discrete[-1:] if len(future_discrete) > 0 else data_window.iloc[-1:, disc_cols_idx].values
            future_discrete = np.tile(last_row, (self.future_seq_len, 1))
        
        # Convert to tensors
        in_seq_continuous = torch.tensor(in_seq_continuous, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
        in_seq_discrete = torch.tensor(in_seq_discrete, dtype=torch.long).unsqueeze(0).to(self.device)
        future_discrete = torch.tensor(future_discrete, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # One-hot encode discrete
        in_seq_discrete_oh = one_hot(in_seq_discrete, config['discrete_dims'], device=self.device)
        future_discrete_oh = one_hot(future_discrete, config['discrete_dims'], device=self.device)
        
        return in_seq_continuous, in_seq_discrete_oh, future_discrete_oh
    
    def _apply_calibration(self, predictions_norm, volatility_norm, timeframe, steps=1):
        """
        Apply dynamic calibration to predictions.
        
        Args:
            predictions_norm: Array of shape (future_seq_len, 3) - normalized predictions
            volatility_norm: Normalized volatility value
            timeframe: '15min', '4hr', '1day', or '1week'
            steps: Number of steps to calibrate (1 or 2)
        
        Returns:
            Calibrated predictions (normalized) of shape (steps, 3)
        """
        calib = self.calibration_params[timeframe]
        if calib is None:
            return predictions_norm[:steps] if predictions_norm.shape[0] > steps else predictions_norm
        
        # Get base offsets
        base_delta_lower_norm = calib['base_delta_lower_norm']
        base_delta_upper_norm = calib['base_delta_upper_norm']
        volatility_mean_norm = calib['volatility_mean_norm']
        volatility_std_norm = calib['volatility_std_norm']
        min_offset_scale = calib['min_offset_scale']
        max_offset_scale = calib['max_offset_scale']
        
        # Compute volatility scaling
        vol_normalized = (volatility_norm - volatility_mean_norm) / volatility_std_norm
        vol_scale = np.clip(vol_normalized, min_offset_scale, max_offset_scale)
        
        # Apply dynamic offsets to each step
        calibrated = []
        for step_idx in range(steps):
            if step_idx >= predictions_norm.shape[0]:
                break
            delta_lower = base_delta_lower_norm * vol_scale
            delta_upper = base_delta_upper_norm * vol_scale
            
            pred_lower = predictions_norm[step_idx, 0] - delta_lower
            pred_median = predictions_norm[step_idx, 1]
            pred_upper = predictions_norm[step_idx, 2] + delta_upper
            
            calibrated.append([pred_lower, pred_median, pred_upper])
        
        return np.array(calibrated)
    
    def predict_at_timestamp(self, timestamp, steps="1", return_raw=False):
        """
        Get predictions from all models at a given timestamp.
        
        MINUTE-CADENCE CONTRACT (Guaranteed):
        - Can be called each minute with timestamp t
        - Always returns the same horizon keys: {15min, 4hr, 1day, 1week}
        - Each horizon provides return quantiles (q10, q50, q90) + last_close
        - steps='1': Returns 1-step ahead predictions (shape: [1] per quantile)
        - steps='2': Returns 2-step ahead predictions (shape: [2] per quantile)
        
        Args:
            timestamp: str or pd.Timestamp - prediction time (bar timestamp)
            steps: "1" or "2" - number of prediction steps to return
                   "1" = single step ahead (default)
                   "2" = both step-0 and step-1 predictions
            return_raw: If True, return raw normalized predictions before calibration
        
        Returns:
            Dictionary with predictions for each timeframe:
            {
                '15min': {
                    'lower': [returns] or [returns_step0, returns_step1],  # 10% quantile
                    'median': [returns] or [returns_step0, returns_step1], # 50% quantile
                    'upper': [returns] or [returns_step0, returns_step1],  # 90% quantile
                    'last_close': float,  # Last close price (USD)
                    'timestamp': pd.Timestamp,  # Prediction timestamp
                    'timeframe': str,  # '15min', '4hr', '1day', or '1week'
                    'metadata': {
                        'bar_timestamp': pd.Timestamp,  # Requested prediction timestamp
                        'data_last_timestamp_used': pd.Timestamp,  # Last data point used
                        'expected_bar_seconds': int,  # Expected bar duration (900 for 15min, etc.)
                        'actual_bar_seconds': float,  # Actual time between last two bars
                        'is_aligned': bool  # Whether bars align with expected duration (within 10% tolerance)
                    }
                },
                '4hr': {...},
                '1day': {...},
                '1week': {...}
            }
            
            When steps="1": quantiles are lists with 1 element (shape: [1])
            When steps="2": quantiles are lists with 2 elements (shape: [2]), one per step
            
            All timeframes use the same last_close reference price (from 1-minute data at timestamp).
        """
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Convert steps to int
        num_steps = int(steps)
        if num_steps not in [1, 2]:
            raise ValueError(f"steps must be '1' or '2', got '{steps}'")
        
        results = {}
        
        for timeframe in ['15min', '4hr', '1day', '1week']:
            try:
                # Resample data up to timestamp
                resampled = self._resample_data(timeframe, timestamp)
                
                # Get timeframe-specific past_seq_len
                config = self.configs[timeframe]
                past_seq_len = config.get('past_seq_len', 80)  # Default 80, override for 1week
                
                if len(resampled) < past_seq_len:
                    print(f"⚠️  {timeframe}: Not enough data ({len(resampled)} < {past_seq_len})")
                    results[timeframe] = None
                    continue
                
                # Get last past_seq_len rows
                data_window = resampled.iloc[-past_seq_len:].copy()
                
                # Prepare input
                in_seq_cont, in_seq_disc, future_disc = self._prepare_input(data_window, timeframe)
                
                # Get volatility for calibration
                feature_idx = {col: i for i, col in enumerate(self.continuous_columns)}
                volatility_norm = data_window.iloc[-1][f'volatility_20']
                # Normalize volatility
                vol_stats = self.norm_stats[timeframe]['volatility_20']
                volatility_norm = (volatility_norm - vol_stats.mean) / vol_stats.std
                
                # Run inference
                model = self.models[timeframe]
                model.reset(1, device=self.device)
                
                with torch.no_grad():
                    net_out, _ = model(in_seq_cont, in_seq_disc, None, future_disc)
                    # Extract requested number of steps: shape (batch, future_seq_len, 3)
                    # net_out shape: (1, 2, 3) -> extract first num_steps steps
                    predictions_norm = net_out[:, :num_steps, :].cpu().numpy()  # Shape: (1, num_steps, 3)
                    predictions_norm = predictions_norm[0]  # Remove batch dimension: (num_steps, 3)
                
                # Handle quantile ordering for each step
                predictions_norm_ordered = []
                for step_idx in range(num_steps):
                    if step_idx >= predictions_norm.shape[0]:
                        # Pad with last step if needed
                        step_preds = predictions_norm[-1]
                    else:
                        step_preds = predictions_norm[step_idx]
                    
                    # Ensure quantile ordering (q10 <= q50 <= q90)
                    if step_preds[0] > step_preds[1]:
                        # Reorder if needed
                        pred_lower_norm = step_preds[2]
                        pred_median_norm = step_preds[1]
                        pred_upper_norm = step_preds[0]
                    else:
                        pred_lower_norm = step_preds[0]
                        pred_median_norm = step_preds[1]
                        pred_upper_norm = step_preds[2]
                    
                    predictions_norm_ordered.append([pred_lower_norm, pred_median_norm, pred_upper_norm])
                
                predictions_norm_ordered = np.array(predictions_norm_ordered)  # Shape: (num_steps, 3)
                
                # Apply calibration
                if not return_raw:
                    predictions_norm_ordered = self._apply_calibration(
                        predictions_norm_ordered, volatility_norm, timeframe, steps=num_steps
                    )
                
                # Denormalize
                ret_stats = self.norm_stats[timeframe]['returns']
                close_stats = self.norm_stats[timeframe]['Close']
                
                # Denormalize all steps
                pred_lower_ret = []
                pred_median_ret = []
                pred_upper_ret = []
                
                for step_idx in range(num_steps):
                    pred_lower_ret.append(predictions_norm_ordered[step_idx, 0] * ret_stats.std + ret_stats.mean)
                    pred_median_ret.append(predictions_norm_ordered[step_idx, 1] * ret_stats.std + ret_stats.mean)
                    pred_upper_ret.append(predictions_norm_ordered[step_idx, 2] * ret_stats.std + ret_stats.mean)
                
                last_close = data_window.iloc[-1]['Close']
                
                # Extract forecast integrity metadata
                bar_timestamp = timestamp
                data_last_timestamp_used = pd.to_datetime(data_window.iloc[-1]['timestamp'])
                expected_bar_seconds = self.expected_bar_seconds[timeframe]
                
                # Calculate actual bar seconds (time difference between last two bars)
                if len(data_window) >= 2:
                    last_bar_time = pd.to_datetime(data_window.iloc[-1]['timestamp'])
                    prev_bar_time = pd.to_datetime(data_window.iloc[-2]['timestamp'])
                    actual_bar_seconds = (last_bar_time - prev_bar_time).total_seconds()
                else:
                    actual_bar_seconds = expected_bar_seconds  # Default to expected if only one bar
                
                # Check alignment: allow some tolerance (e.g., within 10% of expected)
                tolerance = 0.1
                is_aligned = abs(actual_bar_seconds - expected_bar_seconds) / expected_bar_seconds <= tolerance
                
                # Return format: if steps=1, return [value]; if steps=2, return [value_step0, value_step1]
                results[timeframe] = {
                    'lower': pred_lower_ret,  # Returns (percent) - list of length num_steps
                    'median': pred_median_ret,
                    'upper': pred_upper_ret,
                    'last_close': float(last_close),  # USD
                    'timestamp': timestamp,
                    'timeframe': timeframe,
                    # Forecast integrity metadata
                    'metadata': {
                        'bar_timestamp': bar_timestamp,
                        'data_last_timestamp_used': data_last_timestamp_used,
                        'expected_bar_seconds': expected_bar_seconds,
                        'actual_bar_seconds': float(actual_bar_seconds),
                        'is_aligned': bool(is_aligned)
                    }
                }
                
            except Exception as e:
                print(f"⚠️  Error predicting {timeframe}: {e}")
                results[timeframe] = None
        
        return results


def main():
    """Example usage."""
    predictor = MultiTimeframePredictor()
    
    # Example: Predict at a specific timestamp
    timestamp = '2024-11-20 12:00:00'
    print(f"\nPredicting at timestamp: {timestamp}")
    print("=" * 70)
    
    predictions = predictor.predict_at_timestamp(timestamp)
    
    for timeframe, pred in predictions.items():
        if pred is None:
            print(f"\n{timeframe}: No prediction available")
            continue
        
        print(f"\n{timeframe.upper()} Predictions:")
        print(f"  Last Close: ${pred['last_close']:,.2f}")
        print(f"  Lower (10%): {pred['lower'][0]*100:.4f}% → ${pred['last_close']*(1+pred['lower'][0]):,.2f}")
        print(f"  Median (50%): {pred['median'][0]*100:.4f}% → ${pred['last_close']*(1+pred['median'][0]):,.2f}")
        print(f"  Upper (90%): {pred['upper'][0]*100:.4f}% → ${pred['last_close']*(1+pred['upper'][0]):,.2f}")
        print(f"  Band Width: {(pred['upper'][0] - pred['lower'][0])*100:.4f}%")


if __name__ == "__main__":
    main()

