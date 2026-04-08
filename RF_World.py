"""
RF_Market_WorldModel.py
World Model Training Pipeline for Alpha Go Everywhere Replication.

Based on Section 3.2 of Choi, Jiang, Zhang (2025):
- Pools all stocks from all 32 markets together
- Adds 31 country dummies (USA as baseline) as additional features
- Trains a single unified RF model on pooled data
- Tests on each market's test period separately

Hyperparameters from the paper:
- max_depth: [2, 4, 6]
- max_features: [3, 5, 10]
- n_estimators: 300
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.metrics import mean_squared_error
from joblib import dump, load

# =============================================================================
# CONFIGURATION - UPDATE THIS PATH TO YOUR PROJECT FOLDER
# =============================================================================

base = Path(r'C:\Users\matle\Desktop\VSCode\factor_model_project')

# Import from your existing RF_Load_Data
import sys
sys.path.insert(0, str(base))
from RF_Load_Data import load_data, in_output, split_data, load_markets, MARKET_INFO

YOUR_PATH = str(base)

# Hyperparameter grid from the paper
PARAM_GRID = {
    'max_depth': [2, 4, 6],
    'max_features': [3, 5, 10],
}
N_ESTIMATORS = 300


# =============================================================================
# DATA POOLING FUNCTIONS
# =============================================================================

def load_all_markets_data():
    """
    Load data from all 32 markets and return as a dictionary.
    """
    markets = list(MARKET_INFO.keys())
    all_data = {}
    
    print("Loading data from all markets...")
    for market in markets:
        try:
            data, start, train, valid, end, batch = load_data(YOUR_PATH, market)
            all_data[market] = {
                'data': data,
                'start': start,
                'train': train,
                'valid': valid,
                'end': end
            }
            print(f"  {market}: {len(data):,} observations")
        except Exception as e:
            print(f"  {market}: FAILED - {e}")
    
    print(f"\nLoaded {len(all_data)} markets successfully")
    return all_data


def create_pooled_dataset(all_data, add_year):
    """
    Create pooled training, validation, and test datasets from all markets.
    
    For the World Model:
    - Training: Pool all markets' training data for year (valid_split + add_year)
    - Validation: Pool all markets' validation data
    - Test: Each market's test data (kept separate for evaluation)
    
    Adds 31 country dummies (USA as baseline).
    """
    train_dfs = []
    valid_dfs = []
    test_dfs = {}  # Keep test data separate by market
    
    # Get list of markets (USA will be baseline for dummies)
    markets = sorted(all_data.keys())
    non_usa_markets = [m for m in markets if m != 'USA']
    
    print(f"\n  Creating pooled data for add_year={add_year}...")
    
    for market in markets:
        market_info = all_data[market]
        data = market_info['data'].copy()
        train_split = market_info['train']
        valid_split = market_info['valid']
        end_year = market_info['end']
        
        # Check if this year is within market's range
        cur_year = valid_split + add_year
        if cur_year > end_year:
            continue
        
        # Split data using your existing function
        try:
            train_data, valid_data, test_data = split_data(data, train_split, valid_split, add_year)
        except Exception as e:
            print(f"    {market}: Split failed - {e}")
            continue
        
        # Add market identifier column
        train_data['MARKET'] = market
        valid_data['MARKET'] = market
        test_data['MARKET'] = market
        
        # Add country dummies (USA is baseline, so we add dummies for non-USA markets)
        for m in non_usa_markets:
            train_data[f'dummy_{m}'] = (train_data['MARKET'] == m).astype(int)
            valid_data[f'dummy_{m}'] = (valid_data['MARKET'] == m).astype(int)
            test_data[f'dummy_{m}'] = (test_data['MARKET'] == m).astype(int)
        
        if len(train_data) > 0:
            train_dfs.append(train_data)
        if len(valid_data) > 0:
            valid_dfs.append(valid_data)
        if len(test_data) > 0:
            test_dfs[market] = test_data
    
    if not train_dfs or not valid_dfs:
        return None, None, None
    
    pooled_train = pd.concat(train_dfs, ignore_index=True)
    pooled_valid = pd.concat(valid_dfs, ignore_index=True)
    
    print(f"    Pooled train: {len(pooled_train):,} obs from {len(train_dfs)} markets")
    print(f"    Pooled valid: {len(pooled_valid):,} obs from {len(valid_dfs)} markets")
    print(f"    Test markets: {len(test_dfs)}")
    
    return pooled_train, pooled_valid, test_dfs


def in_output_world(data):
    """
    Split data into input features (X) and output target (Y).
    Includes country dummies in features.
    """
    exclude_cols = ['PERMNO', 'DATE', 'TARGET', 'MARKET', 'date_bk', 'permno', 'date', 'target']
    feature_cols = [c for c in data.columns if c not in exclude_cols]
    
    X = data[feature_cols]
    Y = data['TARGET']
    
    return X, Y


# =============================================================================
# WORLD MODEL TRAINING
# =============================================================================

def train_world_rf_year(add_year, all_data):
    """
    Train World Model Random Forest for a specific year.
    """
    # Determine the reference year (using USA's valid_split as reference)
    usa_valid = all_data['USA']['valid']
    cur_year = usa_valid + add_year
    
    print(f"\n  Year {cur_year}: ", end="", flush=True)
    start_time = time.time()
    
    # Create output directories
    params_dir = base / 'model_parameters' / 'WORLD' / 'rf'
    os.makedirs(params_dir, exist_ok=True)
    
    # Create pooled datasets
    pooled_train, pooled_valid, test_dfs = create_pooled_dataset(all_data, add_year)
    
    if pooled_train is None or len(pooled_train) == 0:
        print("Skipping - no pooled training data")
        return None
    
    # Separate features and target
    train_X, train_y = in_output_world(pooled_train)
    valid_X, valid_y = in_output_world(pooled_valid)
    
    # Convert to numpy arrays
    train_X = np.asarray(train_X)
    valid_X = np.asarray(valid_X)
    train_y = np.asarray(train_y)
    valid_y = np.asarray(valid_y)
    
    n_features = train_X.shape[1]
    
    # Cap max_features at number of available features
    param_grid = {
        'max_depth': PARAM_GRID['max_depth'],
        'max_features': [min(mf, n_features) for mf in PARAM_GRID['max_features']]
    }
    param_grid['max_features'] = sorted(list(set(param_grid['max_features'])))
    
    print(f"Features={n_features} ", end="")
    
    # Hyperparameter tuning
    best_mse = np.inf
    best_params = None
    best_model = None
    
    for max_depth in param_grid['max_depth']:
        for max_features in param_grid['max_features']:
            rf = RF(
                n_estimators=N_ESTIMATORS,
                max_depth=max_depth,
                max_features=max_features,
                n_jobs=-1,
                random_state=42
            )
            
            rf.fit(train_X, train_y)
            valid_pred = rf.predict(valid_X)
            mse = np.mean((valid_y - valid_pred) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_params = {'max_depth': max_depth, 'max_features': max_features}
                best_model = rf
    
    print(f"depth={best_params['max_depth']}, feat={best_params['max_features']} ", end="")
    
    # Save model
    model_path = params_dir / f'year{cur_year}.joblib'
    dump(best_model, model_path)
    
    # Get feature names for importance tracking
    sample_X, _ = in_output_world(pooled_train.head())
    feature_names = list(sample_X.columns)
    
    # Generate predictions for each market's test data
    all_predictions = {}
    
    for market, test_data in test_dfs.items():
        if len(test_data) == 0:
            continue
        
        test_data = test_data.reset_index(drop=True)
        test_X, test_y = in_output_world(test_data)
        test_X = np.asarray(test_X)
        
        test_pred = best_model.predict(test_X)
        
        y_pred_test = pd.concat([
            test_data[['PERMNO', 'DATE', 'TARGET', 'MARKET']],
            pd.DataFrame(test_pred, columns=['pred'])
        ], axis=1)
        
        all_predictions[market] = y_pred_test
    
    # Save predictions by market
    for market, pred_df in all_predictions.items():
        forecasts_dir = base / 'forecasts_WorldModel' / market / 'rf' / f'Year{cur_year}'
        os.makedirs(forecasts_dir, exist_ok=True)
        pred_df.to_csv(forecasts_dir / 'test_pred.csv', index=False)
    
    elapsed = time.time() - start_time
    print(f"({elapsed:.1f}s)", flush=True)
    
    return {
        'year': cur_year,
        'model': best_model,
        'params': best_params,
        'feature_names': feature_names,
        'predictions': all_predictions
    }


def train_world_rf_all_years(all_data):
    """
    Train World Model for all applicable years.
    """
    # Determine year range (use USA's range as reference)
    usa_info = all_data['USA']
    valid_split = usa_info['valid']
    end_year = usa_info['end']
    
    add_years = range(1, end_year + 1 - valid_split)
    
    print(f"\nTraining World Model for years {valid_split + 1} to {end_year}")
    print(f"Total years to train: {len(add_years)}")
    
    all_results = []
    all_feature_importances = []
    
    for add_year in add_years:
        result = train_world_rf_year(add_year, all_data)
        if result is not None:
            all_results.append(result)
            all_feature_importances.append(result['model'].feature_importances_)
    
    return all_results, np.array(all_feature_importances)


def consolidate_world_predictions():
    """
    Consolidate all World Model predictions by market.
    """
    print("\nConsolidating predictions by market...")
    
    world_dir = base / 'forecasts_WorldModel'
    if not world_dir.exists():
        print("No World Model predictions found")
        return {}
    
    consolidated = {}
    
    for market_dir in sorted(world_dir.iterdir()):
        if not market_dir.is_dir():
            continue
        
        market = market_dir.name
        csvs = list((market_dir / 'rf').glob('*/test_pred.csv'))
        
        if csvs:
            df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
            consolidated[market] = df
            print(f"  {market}: {len(df):,} predictions")
    
    return consolidated


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("ALPHA GO EVERYWHERE - WORLD MODEL (SECTION 3.2)")
    print("Pooled Global Random Forest")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project path: {YOUR_PATH}")
    print()
    
    # Load all markets
    all_data = load_all_markets_data()
    
    if not all_data:
        print("ERROR: No market data loaded. Check data paths.")
        exit(1)
    
    # Train World Model
    start_time = time.time()
    results, feature_importances = train_world_rf_all_years(all_data)
    total_time = time.time() - start_time
    
    print(f"\n{'=' * 70}")
    print(f"WORLD MODEL TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Years trained: {len(results)}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    
    # Save feature importance
    if len(feature_importances) > 0 and len(results) > 0:
        avg_importance = feature_importances.mean(axis=0)
        feature_names = results[0]['feature_names']
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        importance_df['importance_pct'] = (importance_df['importance'] / importance_df['importance'].sum() * 100).round(2)
        
        save_dir = base / 'forecasts_WorldModel'
        os.makedirs(save_dir, exist_ok=True)
        importance_df.to_csv(save_dir / 'rf_world_feature_importance.csv', index=False)
        
        print(f"\nTop 10 Features (World Model):")
        print(importance_df.head(10).to_string(index=False))
        
        # Show country dummy importance
        dummy_features = importance_df[importance_df['feature'].str.startswith('dummy_')]
        if len(dummy_features) > 0:
            print(f"\nCountry Dummy Importance (Top 10):")
            print(dummy_features.head(10).to_string(index=False))
    
    # Consolidate predictions
    consolidated = consolidate_world_predictions()
    
    # Save consolidated predictions
    for market, df in consolidated.items():
        save_path = base / 'forecasts_WorldModel' / market / 'rf_pred.csv'
        df.to_csv(save_path, index=False)
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNext step: Run RF_Metrics_WorldModel.py to calculate metrics")