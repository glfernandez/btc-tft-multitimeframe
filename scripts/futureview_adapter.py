"""
Adapter to convert Ensemble TFT predictions to FutureView format.

This module provides functions to convert the output of MultiTimeframePredictor
(Ensemble TFT) to the FutureView format expected by the MPC pipeline.

Usage:
    from scripts.inference_pipeline import MultiTimeframePredictor
    from scripts.futureview_adapter import ensemble_to_futureview
    
    predictor = MultiTimeframePredictor()
    ensemble_pred = predictor.predict_at_timestamp('2019-01-01 12:30:00')
    
    future_view = ensemble_to_futureview(
        ensemble_pred, 
        current_price=3688.85,
        timestamp='2019-01-01 12:30:00'
    )
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

workspace_root = Path(__file__).parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))


def classify_trend(net_move_pct: float, threshold: float = 0.005) -> str:
    """
    Classify trend based on net move percentage.
    
    Args:
        net_move_pct: Net move percentage (e.g., 0.01 = 1%)
        threshold: Threshold for flat classification
    
    Returns:
        'up', 'down', or 'flat'
    """
    if net_move_pct > threshold:
        return 'up'
    elif net_move_pct < -threshold:
        return 'down'
    else:
        return 'flat'


def compute_alignment_score(trends: list) -> float:
    """
    Compute alignment score from trend classifications.
    
    Args:
        trends: List of trend strings ['up'/'down'/'flat', ...]
    
    Returns:
        Alignment score between 0.0 and 1.0
    """
    if not trends or len(trends) == 0:
        return 0.0
    
    # Count how many agree
    unique_trends = set(trends)
    if len(unique_trends) == 1:
        return 1.0  # All agree
    elif len(unique_trends) == 2:
        # Check if 'flat' is one of them
        if 'flat' in unique_trends:
            non_flat = [t for t in unique_trends if t != 'flat']
            if len(non_flat) == 1:
                # All non-flat agree
                return 0.75
        # Two different trends
        return 0.5
    else:
        # All different
        return 0.0


def estimate_volatility_from_quantiles(lower_ret: float, upper_ret: float) -> float:
    """
    Estimate volatility from quantile range.
    
    Uses the assumption that for a normal distribution:
    - 80% coverage (10% to 90% quantiles) ≈ 2.56 standard deviations
    
    Args:
        lower_ret: Lower quantile (10%) return
        upper_ret: Upper quantile (90%) return
    
    Returns:
        Estimated volatility (standard deviation)
    """
    if upper_ret == lower_ret:
        return 0.0
    
    # 80% coverage ≈ 2.56 std devs for normal distribution
    quantile_range = upper_ret - lower_ret
    estimated_vol = quantile_range / 2.56
    return max(0.0, estimated_vol)


def ensemble_to_futureview(
    ensemble_predictions: Dict[str, Any],
    current_price: float,
    timestamp: pd.Timestamp,
    compute_volatility: bool = True
) -> Dict[str, Any]:
    """
    Convert Ensemble TFT predictions to FutureView format.
    
    This function maps the output structure from MultiTimeframePredictor
    to the format expected by the MPC pipeline (FutureView object fields).
    
    Args:
        ensemble_predictions: Dictionary from MultiTimeframePredictor.predict_at_timestamp()
            Format: {
                '15min': {'lower': [ret], 'median': [ret], 'upper': [ret], 'last_close': price, ...},
                '4hr': {...},
                '1day': {...},
                '1week': {...}
            }
        current_price: Current market price at timestamp
        timestamp: Prediction timestamp
        compute_volatility: If True, estimate volatility from quantiles
    
    Returns:
        Dictionary with FutureView-compatible fields:
        {
            'max_future_up_pct_15m': float,
            'max_future_drawdown_pct_15m': float,
            'net_future_move_pct_15m': float,
            'median_future_price_pct_15m': float,
            'vol_15m': float,
            # ... same for 4h, 1d, 1w ...
            'short_trend': str,
            'mid_trend': str,
            'long_trend': str,
            'alignment_score': float,
            ...
        }
    """
    # Convert timestamp to pd.Timestamp if needed
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
    
    # Timeframe mapping: TFT names → FutureView suffixes
    tf_map = {
        '15min': '15m',
        '4hr': '4h',
        '1day': '1d',
        '1week': '1w'
    }
    
    fv = {}
    trends = {}
    
    # Process each timeframe
    for tf_tft, tf_fv in tf_map.items():
        pred = ensemble_predictions.get(tf_tft)
        
        if pred is None:
            # Missing prediction - set to zeros
            fv[f'max_future_up_pct_{tf_fv}'] = 0.0
            fv[f'max_future_drawdown_pct_{tf_fv}'] = 0.0
            fv[f'net_future_move_pct_{tf_fv}'] = 0.0
            fv[f'median_future_price_pct_{tf_fv}'] = 0.0
            fv[f'vol_{tf_fv}'] = 0.0
            continue
        
        # Extract quantiles (returns as decimals, e.g., 0.01 = 1%)
        lower_ret = pred['lower'][0] if isinstance(pred['lower'], list) else pred['lower']
        median_ret = pred['median'][0] if isinstance(pred['median'], list) else pred['median']
        upper_ret = pred['upper'][0] if isinstance(pred['upper'], list) else pred['upper']
        
        # Get last close price (for reference)
        last_close = pred.get('last_close', current_price)
        
        # Convert returns to percentages (FutureView uses percentages)
        # TFT returns are already in decimal form (0.01 = 1%)
        # FutureView expects percentages (1.0 = 1%)
        
        # Map to FutureView fields
        # net_future_move_pct: Net price change (expected value)
        fv[f'net_future_move_pct_{tf_fv}'] = median_ret * 100.0
        
        # max_future_up_pct: Maximum upside (upper bound)
        # Note: This is a prediction upper bound, not actual max high
        fv[f'max_future_up_pct_{tf_fv}'] = upper_ret * 100.0
        
        # max_future_drawdown_pct: Maximum drawdown (lower bound)
        # Note: This is a prediction lower bound, not actual min low
        fv[f'max_future_drawdown_pct_{tf_fv}'] = lower_ret * 100.0
        
        # median_future_price_pct: Median price change
        fv[f'median_future_price_pct_{tf_fv}'] = median_ret * 100.0
        
        # Volatility: Estimate from quantile range
        if compute_volatility:
            vol_estimate = estimate_volatility_from_quantiles(lower_ret, upper_ret)
            fv[f'vol_{tf_fv}'] = vol_estimate * 100.0  # Convert to percentage
        else:
            fv[f'vol_{tf_fv}'] = 0.0
        
        # Store trend for later
        trends[tf_fv] = classify_trend(median_ret)
    
    # Compute combined flags
    fv['short_trend'] = trends.get('15m', 'flat')
    fv['mid_trend'] = trends.get('4h', 'flat')
    fv['long_trend'] = trends.get('1d', 'flat')
    
    # Compute alignment score
    trend_list = [fv.get('short_trend', 'flat'), 
                  fv.get('mid_trend', 'flat'),
                  fv.get('long_trend', 'flat')]
    fv['alignment_score'] = compute_alignment_score(trend_list)
    
    # Additional flags
    fv['short_bullish_long_bearish'] = (
        fv.get('short_trend') == 'up' and 
        fv.get('long_trend') == 'down'
    )
    
    # Shock risk: Large 15m drawdown but long-term stable
    large_drawdown_15m = fv.get('max_future_drawdown_pct_15m', 0) < -1.0
    long_term_stable = fv.get('long_trend', 'flat') != 'down'
    fv['shock_risk'] = large_drawdown_15m and long_term_stable
    
    # Store metadata
    fv['_timestamp'] = timestamp
    fv['_current_price'] = current_price
    fv['_source'] = 'ensemble_tft'
    
    return fv


def create_futureview_object(
    ensemble_predictions: Dict[str, Any],
    current_price: float,
    timestamp: pd.Timestamp,
    **kwargs
) -> Any:
    """
    Create a FutureView-like object from Ensemble TFT predictions.
    
    This is a convenience function that creates an object with FutureView attributes
    for backward compatibility with existing MPC code.
    
    Args:
        ensemble_predictions: Dictionary from MultiTimeframePredictor
        current_price: Current market price
        timestamp: Prediction timestamp
        **kwargs: Additional arguments passed to ensemble_to_futureview()
    
    Returns:
        SimpleNamespace object with FutureView attributes
    """
    from types import SimpleNamespace
    
    fv_dict = ensemble_to_futureview(
        ensemble_predictions, 
        current_price, 
        timestamp,
        **kwargs
    )
    
    # Create object with attributes
    fv_obj = SimpleNamespace(**fv_dict)
    
    return fv_obj


def extract_mpc_features(ensemble_predictions: Dict[str, Any], current_price: float) -> Dict[str, Dict[str, float]]:
    """
    Extract MPC-ready features from ensemble predictions.
    
    This function computes derived features beyond raw quantiles that are useful
    for MPC optimization, including expected returns, risk measures, and trend indicators.
    
    Args:
        ensemble_predictions: Dictionary from MultiTimeframePredictor.predict_at_timestamp()
        current_price: Current market price at timestamp
    
    Returns:
        Dictionary with features for each timeframe:
        {
            '15min': {
                'expected_return': float,      # q50
                'upside_risk': float,          # q90 - q50
                'downside_risk': float,        # q50 - q10
                'prediction_range': float,     # q90 - q10 (total uncertainty)
                'asymmetry': float,            # (q90 - q50) - (q50 - q10) (skewness)
                'confidence': float,          # Inverse of range (higher = more confident)
                'trend_direction': int,        # 1 if q50 > 0, else -1
                'trend_strength': float,       # abs(q50)
                'q10': float,
                'q50': float,
                'q90': float,
                'expected_price': float,
                'upper_price': float,
                'lower_price': float,
            },
            '4hr': {...},
            '1day': {...},
            '1week': {...}
        }
    """
    features = {}
    
    for tf in ['15min', '4hr', '1day', '1week']:
        pred = ensemble_predictions.get(tf)
        if pred is None:
            continue
        
        # Extract quantiles (returns as decimals)
        q10 = pred['lower'][0] if isinstance(pred['lower'], list) else pred['lower']
        q50 = pred['median'][0] if isinstance(pred['median'], list) else pred['median']
        q90 = pred['upper'][0] if isinstance(pred['upper'], list) else pred['upper']
        
        # Compute requested features
        expected_return = q50
        upside_risk = q90 - q50
        downside_risk = q50 - q10
        
        # Additional useful features
        prediction_range = q90 - q10
        asymmetry = (q90 - q50) - (q50 - q10)  # Positive = right-skewed, negative = left-skewed
        confidence = 1.0 / (prediction_range + 1e-8)  # Higher = more confident (narrower range)
        
        # Trend indicators
        trend_direction = 1 if q50 > 0 else -1
        trend_strength = abs(q50)
        
        # Price predictions
        expected_price = current_price * (1 + q50)
        upper_price = current_price * (1 + q90)
        lower_price = current_price * (1 + q10)
        
        features[tf] = {
            # Requested features
            'expected_return': expected_return,
            'upside_risk': upside_risk,
            'downside_risk': downside_risk,
            
            # Additional features
            'prediction_range': prediction_range,
            'asymmetry': asymmetry,
            'confidence': confidence,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            
            # Raw quantiles (for reference)
            'q10': q10,
            'q50': q50,
            'q90': q90,
            
            # Price predictions
            'expected_price': expected_price,
            'upper_price': upper_price,
            'lower_price': lower_price,
        }
    
    return features


def compute_trend_score_multi_step(predictions_2step: np.ndarray) -> float:
    """
    Compute trend score from 2-step predictions.
    
    This function computes the slope/trend across multiple prediction steps.
    Currently, the system returns only 1-step ahead, but this function is provided
    for future use when multi-step predictions are enabled.
    
    Args:
        predictions_2step: Array of shape (2, 3) - 2 steps, 3 quantiles (q10, q50, q90)
    
    Returns:
        trend_score: Slope across steps (positive = upward trend, negative = downward)
    """
    if predictions_2step.shape[0] < 2:
        # Only 1 step available, return trend based on sign
        return predictions_2step[0, 1]  # q50 of first step
    
    step1_median = predictions_2step[0, 1]  # q50 at step 1
    step2_median = predictions_2step[1, 1]  # q50 at step 2
    
    # Trend = slope between steps
    trend_score = step2_median - step1_median
    
    return trend_score


# Example usage and testing
if __name__ == "__main__":
    from scripts.inference_pipeline import MultiTimeframePredictor
    
    print("=" * 70)
    print("FutureView Adapter Test")
    print("=" * 70)
    print()
    
    # Initialize predictor
    print("Initializing Ensemble TFT predictor...")
    predictor = MultiTimeframePredictor()
    print()
    
    # Get predictions
    timestamp = '2019-01-01 12:30:00'
    current_price = 3688.85
    
    print(f"Getting predictions at: {timestamp}")
    ensemble_pred = predictor.predict_at_timestamp(timestamp)
    print()
    
    # Convert to FutureView format
    print("Converting to FutureView format...")
    future_view = ensemble_to_futureview(ensemble_pred, current_price, timestamp)
    print()
    
    # Display results
    print("=" * 70)
    print("FutureView-Compatible Output:")
    print("=" * 70)
    print()
    
    for tf in ['15m', '4h', '1d', '1w']:
        print(f"{tf.upper()} Horizon:")
        print(f"  Net Move: {future_view[f'net_future_move_pct_{tf}']:.4f}%")
        print(f"  Max Up: {future_view[f'max_future_up_pct_{tf}']:.4f}%")
        print(f"  Max Down: {future_view[f'max_future_drawdown_pct_{tf}']:.4f}%")
        print(f"  Median: {future_view[f'median_future_price_pct_{tf}']:.4f}%")
        print(f"  Volatility: {future_view[f'vol_{tf}']:.4f}%")
        print()
    
    print("Combined Flags:")
    print(f"  Short Trend: {future_view['short_trend']}")
    print(f"  Mid Trend: {future_view['mid_trend']}")
    print(f"  Long Trend: {future_view['long_trend']}")
    print(f"  Alignment Score: {future_view['alignment_score']:.2f}")
    print(f"  Shock Risk: {future_view['shock_risk']}")
    print()
    
    # Extract MPC features
    print("=" * 70)
    print("MPC Features:")
    print("=" * 70)
    print()
    
    mpc_features = extract_mpc_features(ensemble_pred, current_price)
    
    for tf in ['15min', '4hr', '1day', '1week']:
        if tf in mpc_features:
            feat = mpc_features[tf]
            print(f"{tf.upper()} Features:")
            print(f"  Expected Return: {feat['expected_return']*100:.4f}%")
            print(f"  Upside Risk: {feat['upside_risk']*100:.4f}%")
            print(f"  Downside Risk: {feat['downside_risk']*100:.4f}%")
            print(f"  Prediction Range: {feat['prediction_range']*100:.4f}%")
            print(f"  Asymmetry: {feat['asymmetry']*100:.4f}%")
            print(f"  Trend Direction: {feat['trend_direction']}")
            print(f"  Trend Strength: {feat['trend_strength']*100:.4f}%")
            print(f"  Expected Price: ${feat['expected_price']:,.2f}")
            print()

