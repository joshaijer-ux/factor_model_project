"""
Load_Data.py
Data loading and preprocessing for Alpha Go Everywhere replication.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# =============================================================================
# CONFIGURATION - RELATIVE PATHS
# =============================================================================

base = Path(__file__).parent if '__file__' in dir() else Path.cwd()

PROJECT_PATH = str(base)
NORMALIZED_PATH = str(base / "normalized")
PROCESSED_PATH = str(base / "processed_countries")

os.makedirs(str(base / "model_parameters"), exist_ok=True)
os.makedirs(str(base / "forecasts"), exist_ok=True)

# =============================================================================
# MARKET INFORMATION
# =============================================================================

MARKET_INFO = {
    'USA': {'start': 1970, 'train': 1987, 'valid': 1997, 'end': 2024, 'batch_size': 10000},
    'CAN': {'start': 1980, 'train': 1995, 'valid': 2005, 'end': 2024, 'batch_size': 2000},
    'GBR': {'start': 1980, 'train': 1995, 'valid': 2005, 'end': 2024, 'batch_size': 2000},
    'DEU': {'start': 1980, 'train': 1995, 'valid': 2005, 'end': 2024, 'batch_size': 2000},
    'FRA': {'start': 1980, 'train': 1995, 'valid': 2005, 'end': 2024, 'batch_size': 2000},
    'JPN': {'start': 1980, 'train': 1995, 'valid': 2005, 'end': 2024, 'batch_size': 5000},
    'AUS': {'start': 1985, 'train': 2000, 'valid': 2010, 'end': 2024, 'batch_size': 2000},
    'CHN': {'start': 1995, 'train': 2005, 'valid': 2012, 'end': 2024, 'batch_size': 3000},
}


def load_markets(your_path=None):
    """Load list of available markets from the normalized folder."""
    markets = []
    
    if os.path.exists(NORMALIZED_PATH):
        for f in os.listdir(NORMALIZED_PATH):
            if f.endswith('_ranked.parquet'):
                market = f.replace('_ranked.parquet', '').upper()
                markets.append(market)
    
    if not markets and os.path.exists(PROCESSED_PATH):
        for f in os.listdir(PROCESSED_PATH):
            if f.endswith('.parquet') or f.endswith('.csv'):
                market = f.split('.')[0].split('_')[0].upper()
                if market not in markets:
                    markets.append(market)
    
    if not markets:
        markets = list(MARKET_INFO.keys())
    
    print(f"Available markets: {markets}")
    return sorted(markets)


def load_data(your_path, market):
    """Load data for a specific market."""
    market = market.upper()
    
    possible_paths = [
        str(base / "normalized" / f'{market}_ranked.parquet'),
        str(base / "normalized" / f'{market.lower()}_ranked.parquet'),
        str(base / "processed_countries" / f'{market}.parquet'),
        str(base / "processed_countries" / f'{market}_ranked.parquet'),
        str(base / "processed_countries" / f'{market}.csv'),
    ]
    
    final_data = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            if path.endswith('.parquet'):
                final_data = pd.read_parquet(path)
            else:
                final_data = pd.read_csv(path)
            break
    
    if final_data is None:
        raise FileNotFoundError(f"Could not find data file for market: {market}")
    
    # Standardize column names
    final_data.columns = [c.upper() if c.upper() in ['PERMNO', 'DATE', 'TARGET'] 
                          else c.lower() for c in final_data.columns]
    
    # Ensure DATE is datetime
    final_data['DATE'] = pd.to_datetime(final_data['DATE'])
    
    
    # If PERMNO is all NaN, create sequential IDs
    if final_data['PERMNO'].isna().all():
        print(f"  Warning: PERMNO is empty for {market}, creating sequential IDs")
        final_data['PERMNO'] = range(len(final_data))
    
    # Get market-specific parameters
    if market in MARKET_INFO:
        info = MARKET_INFO[market]
        start_year = info['start']
        train_split = info['train']
        valid_split = info['valid']
        end_year = info['end']
        batch_size = info['batch_size']
    else:
        years = final_data['DATE'].dt.year
        start_year = int(years.min())
        end_year = int(years.max())
        total_years = end_year - start_year
        train_split = start_year + int(total_years * 0.6)
        valid_split = start_year + int(total_years * 0.8)
        batch_size = min(5000, len(final_data) // 10)
        print(f"Using default splits for {market}: train={train_split}, valid={valid_split}, end={end_year}")
    
    # Clean data - only drop NaN in feature columns, not PERMNO/DATE
    feature_cols = [c for c in final_data.columns if c not in ['PERMNO', 'DATE']]
    final_data = final_data.replace([np.inf, -np.inf], np.nan)
    final_data = final_data.dropna(subset=feature_cols, how='any')
    
    # Filter by date range
    final_data = final_data[final_data['DATE'].dt.year <= end_year]
    final_data = final_data[final_data['DATE'].dt.year > start_year]
    final_data.reset_index(drop=True, inplace=True)
    
    print(f"Market: {market}")
    print(f"  Date range: {start_year} - {end_year}")
    print(f"  Train/Valid/Test splits: {train_split}/{valid_split}/{end_year}")
    print(f"  Observations: {len(final_data):,}")
    print(f"  Features: {final_data.shape[1] - 3}")
    
    return final_data, start_year, train_split, valid_split, end_year, batch_size


def in_output(data, year=None, market=None):
    """Split data into input features (X) and output target (Y)."""
    exclude_cols = ['PERMNO', 'DATE', 'TARGET', 'date_bk', 'permno', 'date', 'target']
    feature_cols = [c for c in data.columns if c not in exclude_cols]
    
    X = data[feature_cols]
    Y = data['TARGET']
    
    return X, Y


def split_data(final_data, train_split, valid_split, add_year):
    """Split data into train, validation, and test sets using expanding window."""
    
    train_end_year = train_split + add_year
    valid_end_year = valid_split + add_year
    test_end_year = valid_split + add_year + 1
    
    train_data = final_data[final_data['DATE'].dt.year < train_end_year].copy()
    
    valid_data = final_data[
        (final_data['DATE'].dt.year >= train_end_year) & 
        (final_data['DATE'].dt.year < valid_end_year)
    ].copy()
    
    test_data = final_data[
        (final_data['DATE'].dt.year >= valid_end_year) & 
        (final_data['DATE'].dt.year < test_end_year)
    ].copy()
    
    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    
    print(f"  Split (year {valid_split + add_year}):")
    print(f"    Train: {len(train_data):,} obs")
    print(f"    Valid: {len(valid_data):,} obs")
    print(f"    Test:  {len(test_data):,} obs")
    
    return train_data, valid_data, test_data


def get_feature_names(data):
    """Get list of feature column names."""
    exclude = ['PERMNO', 'DATE', 'TARGET', 'date_bk', 'permno', 'date', 'target']
    return [c for c in data.columns if c not in exclude]


if __name__ == '__main__':
    print("Testing Load_Data module...\n")
    print(f"Project path: {PROJECT_PATH}")
    print(f"Normalized path: {NORMALIZED_PATH}\n")
    
    markets = load_markets(PROJECT_PATH)
    
    if markets:
        test_market = 'USA' if 'USA' in markets else markets[0]
        try:
            data, start, train, valid, end, batch = load_data(PROJECT_PATH, test_market)
            train_data, valid_data, test_data = split_data(data, train, valid, 1)
            X, Y = in_output(train_data)
            print(f"\nFeature matrix shape: {X.shape}")
            print(f"Target shape: {Y.shape}")
        except Exception as e:
            print(f"Error loading {test_market}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No markets found. Check your data paths.")