"""
RF_Market.py

Hyperparameters:
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

from Load_Data import load_data, in_output, split_data, load_markets

# =============================================================================
# CONFIGURATION - RELATIVE PATHS
# =============================================================================

base = Path(__file__).parent if '__file__' in dir() else Path.cwd()
YOUR_PATH = str(base)

# Hyperparameter grid from the paper
PARAM_GRID = {
    'max_depth': [2, 4, 6],
    'max_features': [3, 5, 10],
}
N_ESTIMATORS = 300


# =============================================================================
# RANDOM FOREST TRAINING
# =============================================================================

def train_rf_year(add_year, market, final_data, train_split, valid_split):
    """
    Train Random Forest for a specific year using expanding window.
    """
    cur_year = valid_split + add_year
    print(f"\n  Year {cur_year}: ", end="", flush=True)
    start_time = time.time()
    
    # Create output directories
    params_dir = str(base / 'model_parameters' / market / 'rf')
    os.makedirs(params_dir, exist_ok=True)
    
    # Split data
    train_data, valid_data, test_data = split_data(final_data, train_split, valid_split, add_year)
    
    # Check if we have enough data
    if len(train_data) == 0 or len(valid_data) == 0 or len(test_data) == 0:
        print(f"Skipping - insufficient data", end="")
        return None, None
    
    # Separate features and target
    train_X, train_y = in_output(train_data)
    valid_X, valid_y = in_output(valid_data)
    test_X, test_y = in_output(test_data)
    
    # Convert to numpy arrays
    train_X = np.asarray(train_X)
    valid_X = np.asarray(valid_X)
    test_X = np.asarray(test_X)
    train_y = np.asarray(train_y)
    valid_y = np.asarray(valid_y)
    
    n_features = train_X.shape[1]
    
    # Cap max_features at number of available features
    param_grid = {
        'max_depth': PARAM_GRID['max_depth'],
        'max_features': [min(mf, n_features) for mf in PARAM_GRID['max_features']]
    }
    param_grid['max_features'] = sorted(list(set(param_grid['max_features'])))
    
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
    
    print(f"depth={best_params['max_depth']}, feat={best_params['max_features']}", end=" ")
    
    # Save model
    model_path = str(base / 'model_parameters' / market / 'rf' / f'year{cur_year}.joblib')
    dump(best_model, model_path)
    
    # Generate predictions
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    
    train_pred = best_model.predict(train_X)
    valid_pred = best_model.predict(valid_X)
    test_pred = best_model.predict(test_X)
    
    y_pred_train = pd.concat([
        train_data[['PERMNO', 'DATE', 'TARGET']],
        pd.DataFrame(train_pred, columns=['pred'])
    ], axis=1)
    
    y_pred_valid = pd.concat([
        valid_data[['PERMNO', 'DATE', 'TARGET']],
        pd.DataFrame(valid_pred, columns=['pred'])
    ], axis=1)
    
    y_pred_test = pd.concat([
        test_data[['PERMNO', 'DATE', 'TARGET']],
        pd.DataFrame(test_pred, columns=['pred'])
    ], axis=1)
    
    # Save predictions
    forecasts_dir = str(base / 'forecasts' / market / 'rf' / f'Year{cur_year}')
    os.makedirs(forecasts_dir, exist_ok=True)
    
    y_pred_train.to_csv(str(Path(forecasts_dir) / 'train_pred.csv'), index=False)
    y_pred_valid.to_csv(str(Path(forecasts_dir) / 'valid_pred.csv'), index=False)
    y_pred_test.to_csv(str(Path(forecasts_dir) / 'test_pred.csv'), index=False)
    
    elapsed = time.time() - start_time
    print(f"({elapsed:.1f}s)", end="", flush=True)
    
    return y_pred_test, best_model


def train_rf_years(add_years_list, market, final_data, train_split, valid_split):
    """Train Random Forest for multiple years."""
    predictions = []
    feature_importances = []
    
    for add_year in add_years_list:
        y_pred, model = train_rf_year(add_year, market, final_data, train_split, valid_split)
        if y_pred is not None and model is not None:
            predictions.append(y_pred)
            feature_importances.append(model.feature_importances_)
    
    if not predictions:
        return pd.DataFrame(), np.array([])
    
    return pd.concat(predictions), np.array(feature_importances)


def train_rf_market(market):
    """
    Execute complete Random Forest training for a market.
    """
    print('\n' + '=' * 60)
    print(f"RANDOM FOREST TRAINING: {market}")
    print('=' * 60)
    
    # Load data
    final_data, start_year, train_split, valid_split, end_year, batch_size = load_data(YOUR_PATH, market)
    
    # Check if we have data
    if len(final_data) == 0:
        print(f"No data for {market}, skipping...")
        return None
    
    # Get feature names
    sample_X, _ = in_output(final_data.head())
    feature_names = list(sample_X.columns)
    
    # Test target
    test_y = final_data[final_data['DATE'].dt.year > valid_split][['PERMNO', 'DATE', 'TARGET']].copy()
    test_y.reset_index(drop=True, inplace=True)
    print(f"Test observations: {len(test_y):,}")
    print(f"Features: {len(feature_names)}")
    print(f"Hyperparameter grid: max_depth={PARAM_GRID['max_depth']}, max_features={PARAM_GRID['max_features']}")
    print(f"Trees per forest: {N_ESTIMATORS}")
    
    # Years to train
    add_years = range(1, end_year + 1 - valid_split)
    print(f"Years to train: {valid_split + 1} to {end_year}")
    
    # Train
    start_time = time.time()
    pred_df, feature_imps = train_rf_years(add_years, market, final_data, train_split, valid_split)
    total_time = time.time() - start_time
    
    if pred_df.empty:
        print(f"No predictions generated for {market}")
        return None
    
    # Convert DATE to string for merging
    pred_df['DATE'] = pred_df['DATE'].astype(str)
    test_y['DATE'] = test_y['DATE'].astype(str)
    
    # Merge predictions
    forecast = pd.merge(test_y, pred_df, on=['PERMNO', 'DATE'], how='inner')
    
    if len(forecast) == 0:
        print(f"No matching predictions for {market}")
        return None
    
    # Calculate R² OOS
    ss_res = ((forecast['TARGET'] - forecast['pred']) ** 2).sum()
    ss_tot = ((forecast['TARGET'] - forecast['TARGET'].mean()) ** 2).sum()
    r2_oos = 1 - ss_res / ss_tot
    
    print(f"\n\n{'─' * 40}")
    print(f"RESULTS: {market}")
    print(f"{'─' * 40}")
    print(f"R² OOS: {r2_oos:.4f}")
    print(f"MSE: {ss_res / len(forecast):.6f}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    
    # Save forecast
    save_dir = str(base / 'forecasts' / market)
    os.makedirs(save_dir, exist_ok=True)
    forecast.to_csv(str(Path(save_dir) / 'rf_pred.csv'), index=False)
    
    # Variable importance
    if len(feature_imps) > 0:
        avg_importance = feature_imps.mean(axis=0)
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_importance
        })
        feature_imp = feature_imp.sort_values(by='importance', ascending=False)
        feature_imp['importance_pct'] = (feature_imp['importance'] / feature_imp['importance'].sum() * 100).round(2)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_imp.head(10).to_string(index=False))
        
        feature_imp.to_csv(str(Path(save_dir) / 'rf_feature_importance.csv'), index=False)
    
    return {
        'market': market,
        'r2_oos': r2_oos,
        'mse': ss_res / len(forecast),
        'n_obs': len(forecast),
        'time_minutes': total_time / 60,
        'top_features': feature_imp.head(5)['feature'].tolist() if len(feature_imps) > 0 else []
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("ALPHA GO EVERYWHERE - RANDOM FOREST")
    print("Based on ISLP Chapter 8 Lab")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project path: {YOUR_PATH}")
    print()
    
    markets = load_markets(YOUR_PATH)
    
    if not markets:
        print("ERROR: No markets found. Check data paths.")
        exit(1)
    
    print(f"Available markets: {markets}")
    print()
    
    # =========================================================================
    # CHANGE THIS LINE TO TRAIN SPECIFIC MARKETS
    # =========================================================================
    # Train all markets:
    markets_to_train = markets
    
    # Or train just USA for testing:
    # markets_to_train = ['USA']
    
    # Or train specific markets:
    #markets_to_train = ['VNM', 'ZAF', 'USA']
    # =========================================================================
    
    all_results = []
    
    for market in markets_to_train:
        try:
            result = train_rf_market(market)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"\nERROR training {market}: {e}")
            import traceback
            traceback.print_exc()
    
    if all_results:
        print("\n" + "=" * 60)
        print("SUMMARY: ALL MARKETS")
        print("=" * 60)
        
        summary_df = pd.DataFrame([{
            'Market': r['market'],
            'R² OOS': round(r['r2_oos'], 4),
            'MSE': round(r['mse'], 6),
            'N': r['n_obs'],
            'Time (min)': round(r['time_minutes'], 1)
        } for r in all_results])
        
        print(summary_df.to_string(index=False))
        
        summary_df.to_csv(str(base / 'rf_results_summary.csv'), index=False)
        print(f"\nSummary saved: {str(base / 'rf_results_summary.csv')}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
