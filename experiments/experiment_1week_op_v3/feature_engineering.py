"""
Feature engineering functions for technical indicators.
Creates all features needed for experiment_1week_op_v3.
"""

import pandas as pd
import numpy as np

def calculate_returns(df):
    """Calculate returns and log returns."""
    df = df.copy()
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    return df

def calculate_atr(df, window=10):
    """Calculate Average True Range."""
    df = df.copy()
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f'atr_{window}'] = true_range.rolling(window=window).mean()
    return df

def calculate_volatility(df, window=20):
    """Calculate rolling volatility."""
    df = df.copy()
    df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
    return df

def calculate_rsi(df, window=14):
    """Calculate Relative Strength Index."""
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    return df

def calculate_bollinger_bands(df, window=20, num_std=2):
    """Calculate Bollinger Bands."""
    df = df.copy()
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    
    df['bollinger_upper'] = rolling_mean + (rolling_std * num_std)
    df['bollinger_lower'] = rolling_mean - (rolling_std * num_std)
    df['bollinger_position'] = (df['Close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
    df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / rolling_mean
    return df

def calculate_volume_indicators(df, window=20):
    """Calculate volume-based indicators."""
    df = df.copy()
    df['volume_ma_20'] = df['Volume'].rolling(window=window).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
    
    # On-Balance Volume (OBV)
    df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return df

def calculate_price_position(df, window=20):
    """Calculate price position within rolling window."""
    df = df.copy()
    rolling_min = df['Low'].rolling(window=window).min()
    rolling_max = df['High'].rolling(window=window).max()
    df['price_position'] = (df['Close'] - rolling_min) / (rolling_max - rolling_min)
    return df

def calculate_candlestick_features(df):
    """Calculate candlestick body and wick features."""
    df = df.copy()
    df['body_size'] = np.abs(df['Close'] - df['Open'])
    upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['wick_ratio'] = (upper_wick + lower_wick) / (df['body_size'] + 1e-10)  # Avoid division by zero
    return df

def calculate_momentum(df, periods=[5, 10]):
    """Calculate momentum indicators."""
    df = df.copy()
    for period in periods:
        df[f'momentum_{period}'] = df['Close'].diff(period)
    return df

def calculate_regime_features(df, window=20):
    """Calculate volatility regime features."""
    df = df.copy()
    volatility = df['returns'].rolling(window=window).std()
    df['volatility_regime'] = pd.cut(
        volatility,
        bins=[0, volatility.quantile(0.33), volatility.quantile(0.67), float('inf')],
        labels=[0, 1, 2]  # Low, Medium, High volatility
    ).astype(float)
    df['volatility_percentile'] = volatility.rolling(window=window*5).rank(pct=True)
    return df

def add_time_features(df):
    """Add time-based discrete features (DayOfWeek, DayOfMonth, Month).
    
    Values are adjusted for one-hot encoding:
    - DayOfWeek: 0-6 (7 categories)
    - DayOfMonth: 0-30 (31 categories, 1-31 mapped to 0-30)
    - Month: 0-11 (12 categories, 1-12 mapped to 0-11)
    """
    df = df.copy()
    if 'timestamp' in df.columns:
        df['DayOfWeek'] = pd.to_datetime(df['timestamp']).dt.dayofweek  # 0-6 (Monday=0)
        df['DayOfMonth'] = pd.to_datetime(df['timestamp']).dt.day - 1  # 0-30 (1-31 mapped to 0-30)
        df['Month'] = pd.to_datetime(df['timestamp']).dt.month - 1  # 0-11 (1-12 mapped to 0-11)
    return df

def add_all_features(df):
    """Add all technical indicators to dataframe."""
    df = df.copy()
    
    # Add time features first (for discrete features)
    df = add_time_features(df)
    
    # Calculate in order (some depend on others)
    df = calculate_returns(df)
    df = calculate_atr(df, window=10)
    df = calculate_volatility(df, window=20)
    df = calculate_rsi(df, window=14)
    df = calculate_bollinger_bands(df, window=20, num_std=2)
    df = calculate_volume_indicators(df, window=20)
    df = calculate_price_position(df, window=20)
    df = calculate_candlestick_features(df)
    df = calculate_momentum(df, periods=[5, 10])
    df = calculate_regime_features(df, window=20)
    
    # Fill NaN values with forward fill then backward fill
    df = df.ffill().bfill().fillna(0)
    
    return df


