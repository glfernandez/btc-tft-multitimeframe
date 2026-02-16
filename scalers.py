"""
Column-wise normalization utilities for Temporal Fusion Transformer.

This module provides per-column z-score normalization, replacing the global
normalization approach that caused visualization issues.
"""

from dataclasses import dataclass
import json
import numpy as np
from pathlib import Path
from typing import Dict, List


@dataclass
class ZStats:
    """Statistics for z-score normalization of a single column."""
    mean: float
    std: float


def compute_stats(df, cols: List[str], eps: float = 1e-8) -> Dict[str, ZStats]:
    """
    Compute mean and std for each column in the dataframe.
    
    Args:
        df: DataFrame with columns to normalize
        cols: List of column names to compute stats for
        eps: Minimum std value to avoid division by zero
        
    Returns:
        Dictionary mapping column names to ZStats objects
    """
    stats = {}
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in dataframe")
        mean_val = float(df[c].mean())
        std_val = float(max(df[c].std(), eps))
        stats[c] = ZStats(mean=mean_val, std=std_val)
    return stats


def apply_z(df, stats: Dict[str, ZStats], cols: List[str] = None) -> None:
    """
    Apply z-score normalization to specified columns in-place.
    
    Args:
        df: DataFrame to normalize (modified in-place)
        stats: Dictionary of ZStats for each column
        cols: List of columns to normalize (default: all keys in stats)
    """
    if cols is None:
        cols = list(stats.keys())
    
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in dataframe")
        if c not in stats:
            raise ValueError(f"No stats found for column '{c}'")
        s = stats[c]
        df[c] = (df[c] - s.mean) / s.std


def save_stats(stats: Dict[str, ZStats], path: str) -> None:
    """
    Save normalization statistics to JSON file.
    
    Args:
        stats: Dictionary of ZStats objects
        path: Path to save JSON file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {c: {"mean": s.mean, "std": s.std} for c, s in stats.items()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved normalization stats to: {path}")


def load_stats(path: str) -> Dict[str, ZStats]:
    """
    Load normalization statistics from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Dictionary of ZStats objects
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")
    
    with open(path) as f:
        data = json.load(f)
    
    stats = {c: ZStats(v["mean"], v["std"]) for c, v in data.items()}
    print(f"Loaded normalization stats from: {path}")
    return stats


def inverse_close(z: np.ndarray, stats: Dict[str, ZStats]) -> np.ndarray:
    """
    Inverse transform Close price predictions from normalized to raw scale.
    
    Args:
        z: Normalized Close values (can be array or scalar)
        stats: Dictionary of ZStats (must contain 'Close')
        
    Returns:
        Raw Close prices
    """
    if "Close" not in stats:
        raise ValueError("Stats must contain 'Close' for inverse transform")
    
    m = stats["Close"].mean
    s = stats["Close"].std
    return z * s + m


def validate_stats(stats: Dict[str, ZStats], eps: float = 1e-8) -> None:
    """
    Validate that all stats have non-zero std.
    
    Args:
        stats: Dictionary of ZStats objects
        eps: Minimum acceptable std value
        
    Raises:
        ValueError if any column has std <= eps
    """
    for col, s in stats.items():
        if s.std <= eps:
            raise ValueError(
                f"Column '{col}' has std={s.std:.2e} <= {eps:.2e}. "
                "This will cause division issues. Check for constant columns."
            )


def round_trip_test(df, col: str, stats: Dict[str, ZStats], atol: float = 1e-6) -> bool:
    """
    Test that normalization and inverse transform are exact (round-trip).
    
    Args:
        df: DataFrame with original data
        col: Column name to test
        stats: Dictionary of ZStats
        atol: Absolute tolerance for comparison
        
    Returns:
        True if test passes
        
    Raises:
        AssertionError if round-trip fails
    """
    if col not in stats:
        raise ValueError(f"No stats found for column '{col}'")
    
    x = df[col].to_numpy()
    m = stats[col].mean
    s = stats[col].std
    
    # Normalize
    z = (x - m) / s
    
    # Inverse transform
    x_rt = z * s + m
    
    # Check round-trip
    if not np.allclose(x, x_rt, atol=atol):
        max_diff = np.abs(x - x_rt).max()
        raise AssertionError(
            f"Round-trip test failed for '{col}': max difference = {max_diff:.2e} "
            f"(tolerance = {atol:.2e})"
        )
    
    return True

