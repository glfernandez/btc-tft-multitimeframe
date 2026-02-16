"""
MPC Adapter for Ensemble TFT Predictions

This module provides MPC-ready adapters that convert raw TFT ensemble predictions
into numeric vectors and scenario paths for Model Predictive Control.

Functions:
    - extract_tvp_vector: Convert TFT predictions to numeric tvp_t vector (v1.0)
    - build_minute_path_from_horizons: Build 1-minute price path from multi-horizon predictions
    - build_minute_path_with_bands: Build path with uncertainty bands

TVP Vector Version: v1.0
- Shape: (20,) - 5 features × 4 horizons
- Features per horizon: [expected_return, upside_risk, downside_risk, range, skew_proxy]
- Horizons: [15min, 4hr, 1day, 1week] in order
- Do not reorder without versioning!
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional


def extract_tvp_vector(ensemble_predictions: Dict[str, Any], current_price: float) -> np.ndarray:
    """
    Convert raw TFT ensemble predictions into a numeric tvp_t vector for MPC.
    
    This function extracts MPC-ready features from ensemble predictions, returning
    a fixed-length numeric vector with no nested dicts. The vector includes for
    each horizon (15m, 4h, 1d, 1w):
        - expected_return (q50)
        - upside_risk (q90 - q50)
        - downside_risk (q50 - q10)
        - range (q90 - q10)
        - skew_proxy (asymmetry indicator)
    
    Args:
        ensemble_predictions: Dictionary from MultiTimeframePredictor.predict_at_timestamp()
                            Format: {'15min': {'lower': [float], 'median': [float], 'upper': [float], ...}, ...}
        current_price: Current market price at timestamp
    
    Returns:
        tvp_t: Numeric vector of shape (20,) containing:
            For each of 4 horizons (15m, 4h, 1d, 1w) in order:
                [expected_return, upside_risk, downside_risk, range, skew_proxy]
            Total: 4 horizons × 5 features = 20 elements
        
        Vector structure:
            [15m_expected_return, 15m_upside_risk, 15m_downside_risk, 15m_range, 15m_skew,
             4h_expected_return, 4h_upside_risk, 4h_downside_risk, 4h_range, 4h_skew,
             1d_expected_return, 1d_upside_risk, 1d_downside_risk, 1d_range, 1d_skew,
             1w_expected_return, 1w_upside_risk, 1w_downside_risk, 1w_range, 1w_skew]
    """
    # Feature order: [expected_return, upside_risk, downside_risk, range, skew_proxy]
    features_per_horizon = 5
    horizons = ['15min', '4hr', '1day', '1week']
    tvp_vector = np.zeros(len(horizons) * features_per_horizon, dtype=np.float64)
    
    for idx, tf in enumerate(horizons):
        pred = ensemble_predictions.get(tf)
        if pred is None:
            # Keep zeros if prediction unavailable
            continue
        
        # Extract quantiles (returns as decimals)
        # Handle both single-step [value] and multi-step [value0, value1] formats
        lower = pred['lower']
        median = pred['median']
        upper = pred['upper']
        
        # Use first step if multi-step
        if isinstance(lower, list) and len(lower) > 0:
            q10 = lower[0] if isinstance(lower[0], (int, float)) else lower
            q50 = median[0] if isinstance(median[0], (int, float)) else median
            q90 = upper[0] if isinstance(upper[0], (int, float)) else upper
        else:
            q10 = lower if isinstance(lower, (int, float)) else (lower[0] if isinstance(lower, list) else 0.0)
            q50 = median if isinstance(median, (int, float)) else (median[0] if isinstance(median, list) else 0.0)
            q90 = upper if isinstance(upper, (int, float)) else (upper[0] if isinstance(upper, list) else 0.0)
        
        # Compute features
        expected_return = float(q50)
        upside_risk = float(q90 - q50)
        downside_risk = float(q50 - q10)
        range_val = float(q90 - q10)
        skew_proxy = float((q90 - q50) - (q50 - q10))  # Positive = right-skewed, negative = left-skewed
        
        # Store in vector (5 features per horizon)
        start_idx = idx * features_per_horizon
        tvp_vector[start_idx] = expected_return
        tvp_vector[start_idx + 1] = upside_risk
        tvp_vector[start_idx + 2] = downside_risk
        tvp_vector[start_idx + 3] = range_val
        tvp_vector[start_idx + 4] = skew_proxy
    
    return tvp_vector


def build_minute_path_from_horizons(
    ensemble_predictions: Dict[str, Any],
    current_price: float,
    timestamp: Optional[pd.Timestamp] = None,
    n_minutes: int = 10080,
    use_quantile: str = "median"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build 1-minute price path from multi-horizon ensemble predictions.
    
    Uses piecewise-linear interpolation with anchor points at horizon boundaries.
    Fast and stable for deterministic MPC scenarios.
    
    Args:
        ensemble_predictions: Dictionary from MultiTimeframePredictor.predict_at_timestamp()
                            Must contain '15min', '4hr', '1day', '1week' keys
        current_price: Current market price at timestamp
        timestamp: Optional timestamp for generating time axis (default: None)
        n_minutes: Number of minutes to project (default: 10080 = 1 week)
        use_quantile: Which quantile to use for path ('lower', 'median', 'upper')
                     Default: 'median' for expected path
    
    Returns:
        Tuple of (prices, timestamps):
            - prices: Array of shape (n_minutes,) - absolute prices at each minute
            - timestamps: Array of shape (n_minutes,) - timestamps (if timestamp provided)
                        or array of minute indices (if timestamp=None)
    
    Algorithm:
        1. Extract anchor points at minutes: [0, 15, 240, 1440, 10080]
        2. Convert returns to prices using current_price
        3. Linearly interpolate between anchors to create minute-level path
        
    Example:
        >>> predictor = MultiTimeframePredictor()
        >>> pred = predictor.predict_at_timestamp('2024-01-01 12:00:00')
        >>> prices, times = build_minute_path_from_horizons(pred, current_price=50000.0)
        >>> len(prices)
        10080
        >>> prices[0]
        50000.0
    """
    # Anchor points in minutes: [0, 15, 240, 1440, 10080]
    # Corresponding to: [now, 15min, 4hr, 1day, 1week]
    anchor_times = np.array([0, 15, 240, 1440, 10080], dtype=np.float64)
    
    # Extract quantile values from predictions
    horizons_map = {
        15: '15min',
        240: '4hr',
        1440: '1day',
        10080: '1week'
    }
    
    # Initialize anchor prices with current price
    anchor_prices = [current_price]
    anchor_times_valid = [0.0]
    
    # Build anchor points from predictions
    for anchor_min, tf in horizons_map.items():
        if anchor_min > n_minutes:
            break  # Don't add anchors beyond requested horizon
        
        pred = ensemble_predictions.get(tf)
        if pred is None:
            continue
        
        # Extract quantile value
        quantile_key = use_quantile  # 'lower', 'median', or 'upper'
        quantile_returns = pred.get(quantile_key, pred.get('median', [0.0]))
        
        # Use first step if multi-step
        if isinstance(quantile_returns, list) and len(quantile_returns) > 0:
            return_value = quantile_returns[0]
        else:
            return_value = quantile_returns if isinstance(quantile_returns, (int, float)) else 0.0
        
        # Convert return to price
        anchor_price = current_price * (1.0 + float(return_value))
        anchor_prices.append(anchor_price)
        anchor_times_valid.append(float(anchor_min))
    
    anchor_times_valid = np.array(anchor_times_valid)
    anchor_prices = np.array(anchor_prices)
    
    # Create minute-level time axis
    minute_indices = np.arange(n_minutes, dtype=np.float64)
    
    # Linearly interpolate prices
    prices = np.interp(minute_indices, anchor_times_valid, anchor_prices)
    
    # Generate timestamps if provided
    if timestamp is not None:
        timestamps = pd.date_range(timestamp, periods=n_minutes, freq='1min')
        return prices, timestamps.values
    else:
        return prices, minute_indices


def build_minute_path_with_bands(
    ensemble_predictions: Dict[str, Any],
    current_price: float,
    timestamp: Optional[pd.Timestamp] = None,
    n_minutes: int = 10080
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 1-minute price path with uncertainty bands from multi-horizon predictions.
    
    Returns median path plus lower and upper quantile bands for robust MPC.
    
    Args:
        ensemble_predictions: Dictionary from MultiTimeframePredictor.predict_at_timestamp()
        current_price: Current market price at timestamp
        timestamp: Optional timestamp for generating time axis
        n_minutes: Number of minutes to project (default: 10080 = 1 week)
    
    Returns:
        Tuple of (median_prices, lower_prices, upper_prices, timestamps):
            - median_prices: Array (n_minutes,) - median price path
            - lower_prices: Array (n_minutes,) - q10 price path
            - upper_prices: Array (n_minutes,) - q90 price path
            - timestamps: Array (n_minutes,) - timestamps or indices
    """
    median_prices, timestamps = build_minute_path_from_horizons(
        ensemble_predictions, current_price, timestamp, n_minutes, use_quantile='median'
    )
    lower_prices, _ = build_minute_path_from_horizons(
        ensemble_predictions, current_price, timestamp, n_minutes, use_quantile='lower'
    )
    upper_prices, _ = build_minute_path_from_horizons(
        ensemble_predictions, current_price, timestamp, n_minutes, use_quantile='upper'
    )
    
    return median_prices, lower_prices, upper_prices, timestamps


# Example usage
if __name__ == "__main__":
    from scripts.inference_pipeline import MultiTimeframePredictor
    
    # Initialize predictor
    predictor = MultiTimeframePredictor()
    
    # Get predictions
    timestamp = '2024-01-01 12:00:00'
    predictions = predictor.predict_at_timestamp(timestamp, steps="1")
    
    # Extract tvp vector
    current_price = predictions['15min']['last_close']
    tvp = extract_tvp_vector(predictions, current_price)
    
    print("=" * 70)
    print("MPC Adapter Test")
    print("=" * 70)
    print(f"\nTVP Vector Shape: {tvp.shape}")
    print(f"TVP Vector:\n{tvp}")
    print()
    
    # Build minute path
    prices, times = build_minute_path_from_horizons(predictions, current_price, timestamp, n_minutes=100)
    
    print(f"Price Path Shape: {prices.shape}")
    print(f"First 10 prices: {prices[:10]}")
    print(f"Last 10 prices: {prices[-10:]}")
    print()
    
    # Build path with bands
    median, lower, upper, _ = build_minute_path_with_bands(predictions, current_price, timestamp, n_minutes=100)
    print(f"Median path range: [{median.min():.2f}, {median.max():.2f}]")
    print(f"Lower band range: [{lower.min():.2f}, {lower.max():.2f}]")
    print(f"Upper band range: [{upper.min():.2f}, {upper.max():.2f}]")

