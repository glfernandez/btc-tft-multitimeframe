import pandas as pd
import datetime
import torch
import torch.nn as nn
from statsmodels.tsa.stattools import adfuller
import numpy as np
import time

import random

import plotly.graph_objects as go

header = {'open time' : 1, 'open' : 2, 'high' : 3, 'low' : 4, 'close' : 5}

#load_data from file - adapted to work with custom CSV format
def load_data_from_csv(csv_path, normalise = True):
    """
    Load data from a CSV file with format: timestamp, open, high, low, close, volume
    
    Args:
        csv_path: Path to the CSV file
        normalise: Whether to normalise data (currently not used in this function)
    
    Returns:
        DataFrame with standardized column names and time features
    """
    # Read CSV file
    data = pd.read_csv(csv_path)
    
    # Rename columns to match expected format (capitalize first letter)
    column_mapping = {
        'timestamp': 'timestamp',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    data = data.rename(columns=column_mapping)
    
    # Parse timestamp column (handle both string and numeric timestamps)
    if data['timestamp'].dtype == 'object':
        # String timestamp format
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    else:
        # Assume it's a numeric timestamp (Unix timestamp in seconds)
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    
    # Convert timestamp into time features
    data['Hour'] = data['timestamp'].apply(lambda x: x.hour)
    data['Day'] = data['timestamp'].apply(lambda x: x.day - 1)
    data['Month'] = data['timestamp'].apply(lambda x: x.month - 1)
    
    # Create 'Open Time' column for compatibility (as milliseconds timestamp)
    data['Open Time'] = (data['timestamp'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')
    
    # Ensure numeric columns are numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Sort by timestamp
    data = data.sort_values('timestamp').reset_index(drop=True)
    
    print("Data loaded successfully. Shape:", data.shape)
    return data

#load_data from file (original Binance format - kept for compatibility)
def load_data(year, symbol, con_cols, interval = "1m", normalise = True):
    if(type(year) != type([])):
        year = [year]
    
    if(type(symbol) != type([])):
        symbol = [symbol]
    
    frames = []
    for y in year:
        for s in symbol:
            data = pd.read_csv("data_{}/{}_{}.csv".format(interval, y, s))
            frames.append(data)    
    
    data_ = pd.concat(frames)
    
    print("done")
    #convert timestamp into month and day numbers
    data_['Hour'] = pd.to_datetime(data_["Open Time"], unit='ms').apply(lambda x: x.hour)    
    data_['Day'] = pd.to_datetime(data_["Open Time"], unit='ms').apply(lambda x: x.day - 1)
    data_['Month'] = pd.to_datetime(data_["Open Time"], unit='ms').apply(lambda x: x.month - 1)
    return data_

class Indexer():
    def __init__(self, r_bottom, r_top, batch_size, random = True, increment = 1):
        self.r_bottom = r_bottom
        self.r_top = r_top
        self.random = random
        self.increment = increment
        self.batch_size = batch_size
        self.indices = [0]
        self.next()
        
    def next(self):
        if(self.random):
            new_indices = []
            for b in range(self.batch_size):
                new_indices.append(random.randrange(self.r_bottom, self.r_top))
            self.indices = new_indices
        else:
            new_indices = [self.indices[-1]]
            
            for b in range(1, self.batch_size):
                i = new_indices[-1] + self.increment
                if(i >= self.r_top):
                    new_indices.append((i - self.top) + self.r_bottom)
                else:
                    new_indices.append(i)
            self.indices = new_indices
            
        return self.indices
    
def get_batches(data_, in_seq_len, out_seq_len, con_cols, disc_cols, target_cols, batch_size = 1, device = 'cpu', normalise = True, indexer = None, norm = None, stats = None):
    """
    Generate batches of data for training.
    
    Args:
        data_: Input dataframe
        in_seq_len: Input sequence length
        out_seq_len: Output sequence length
        con_cols: Continuous column names
        disc_cols: Discrete column names
        target_cols: Target column names
        batch_size: Batch size
        device: Device to create tensors on ('cpu', 'cuda', 'mps')
        normalise: Whether to normalize continuous features
        indexer: Custom indexer (optional)
        norm: Data to use for normalization statistics (optional, for backward compatibility)
        stats: Dictionary of ZStats for per-column normalization (optional, preferred method)
    """
    data = data_.copy()
    
    given_indexer = True
    if indexer is None:
        given_indexer = False
        indexer = Indexer(1, data.shape[0] - (in_seq_len + out_seq_len + 1), batch_size)
        
    if normalise:
        if stats is not None:
            # Per-column normalization (new method)
            try:
                from scalers import apply_z
                apply_z(data, stats, cols=con_cols)
            except ImportError:
                raise ImportError("scalers.py not found. Install it or use norm parameter for global normalization.")
        elif norm is not None:
            # Global normalization (legacy method for backward compatibility)
            data[con_cols] = (data[con_cols] - norm[con_cols].stack().mean()) / norm[con_cols].stack().std()
        else:
            # Use data itself for normalization stats
            data[con_cols] = (data[con_cols] - data[con_cols].stack().mean()) / data[con_cols].stack().std()
    
    #convert columns indices from dataframe to numpy darray
    con_cols_idx = [data.columns.get_loc(x) for x in con_cols]
    disc_cols_idx = [data.columns.get_loc(x) for x in disc_cols]
    target_cols_idx = [data.columns.get_loc(x) for x in target_cols]
    
    # Select only numeric columns for tensor conversion
    all_numeric_cols = list(set(con_cols + disc_cols + target_cols))
    numeric_data = data[all_numeric_cols].copy()
    
    # Ensure all columns are numeric
    for col in all_numeric_cols:
        if not pd.api.types.is_numeric_dtype(numeric_data[col]):
            numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
    
    # Update column indices for numeric data
    con_cols = [numeric_data.columns.get_loc(x) for x in con_cols]
    disc_cols = [numeric_data.columns.get_loc(x) for x in disc_cols]
    target_cols = [numeric_data.columns.get_loc(x) for x in target_cols]
        
    while True:
        #get batches
        n = np.array([np.r_[i:(i + in_seq_len + out_seq_len)] for i in indexer.indices])
        batch_data = numeric_data.iloc[n.flatten()].values
        # MPS requires float32, so convert to float32 for all devices for consistency
        batch_data = torch.tensor(batch_data.reshape(batch_size ,in_seq_len + out_seq_len, numeric_data.shape[-1]), 
                                 device=device, dtype=torch.float32)
        
        #split up batch data
        in_seq_continuous = batch_data[:,0:in_seq_len, con_cols]
        in_seq_discrete = batch_data[:,0:in_seq_len, disc_cols]

        out_seq= batch_data[:,in_seq_len:in_seq_len + out_seq_len, disc_cols]
        target_seq = batch_data[:,in_seq_len:in_seq_len + out_seq_len, target_cols]
    
        yield (in_seq_continuous.unsqueeze(-1),
                        in_seq_discrete,
                        out_seq,
                        target_seq)
        
        if(not given_indexer):
            indexer.next()
    
def one_hot(x, dims, device = 'cpu'):
    """
    Convert discrete values to one-hot encoding.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, n_features)
        dims: List of dimensions for each discrete feature
        device: Device to create tensors on ('cpu', 'cuda', 'mps')
    
    Returns:
        List of one-hot encoded tensors
    """
    out = []
    batch_size = x.shape[0]
    seq_len = x.shape[1]
        
    for i in range(0, x.shape[-1]):
        x_ = x[:,:,i].byte().cpu().long().unsqueeze(-1)
        o = torch.zeros([batch_size, seq_len, dims[i]], device=device, dtype=torch.long)
        o.scatter_(-1, x_.to(device), 1)
        out.append(o.float())
    return out