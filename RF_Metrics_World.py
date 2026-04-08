"""
RF_Metrics_WorldModel.py
Calculate all paper metrics for World Model predictions.

Metrics calculated (matching RF_Metrics.py):
- R² OOS (out-of-sample R-squared, per GKX formula)
- MSE
- Rank Correlation (Spearman)
- Sharpe Ratio (Equal-weighted and Value-weighted)
- Decile Score Distance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from joblib import load
import sys

# =============================================================================
# CONFIGURATION - UPDATE THIS PATH TO YOUR PROJECT FOLDER
# =============================================================================

base = Path(r'C:\Users\matle\Desktop\VSCode\factor_model_project')
sys.path.insert(0, str(base))

FEATURE_NAMES = [
    'absacc', 'acc', 'age', 'agr', 'bm', 'bm_ia', 'cashdebt', 'cashpr',
    'cfp', 'cfp_ia', 'chatoia', 'chcsho', 'chempia', 'chinv', 'chpmia',
    'convind', 'currat', 'depr', 'divi', 'divo', 'dy', 'egr', 'ep', 'gma',
    'grcapx', 'grltnoa', 'herf', 'hire', 'invest', 'lev', 'lgr', 'mvel1',
    'operprof', 'orgcap', 'pchcurrat', 'rd'
]


def load_market_data(market):
    """Load original market data to get mvel1 for value-weighting."""
    data_path = base / 'normalized' / f'{market}_ranked.parquet'
    if data_path.exists():
        df = pd.read_parquet(data_path)
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df[['PERMNO', 'DATE', 'mvel1']].copy()
    return None


def get_world_feature_importance():
    """Load World Model feature importance."""
    importance_file = base / 'forecasts_WorldModel' / 'rf_world_feature_importance.csv'
    if importance_file.exists():
        return pd.read_csv(importance_file)
    
    # If no saved file, compute from models
    model_dir = base / 'model_parameters' / 'WORLD' / 'rf'
    if not model_dir.exists():
        return None
    
    model_files = list(model_dir.glob('year*.joblib'))
    if not model_files:
        return None
    
    all_importances = []
    n_features = None
    
    for mf in model_files:
        model = load(mf)
        if n_features is None:
            n_features = model.n_features_in_
        if model.n_features_in_ == n_features:
            all_importances.append(model.feature_importances_)
    
    if not all_importances:
        return None
    
    avg_importance = np.mean(all_importances, axis=0)
    
    # Feature names include original features + 31 country dummies
    feature_names = FEATURE_NAMES[:min(36, n_features)]
    if n_features > 36:
        # Add dummy feature names
        for i in range(n_features - 36):
            feature_names.append(f'dummy_{i}')
    
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(avg_importance)],
        'importance': avg_importance
    }).sort_values('importance', ascending=False)
    importance_df['importance_pct'] = (importance_df['importance'] / importance_df['importance'].sum() * 100).round(2)
    
    return importance_df


# =============================================================================
# MAIN
# =============================================================================

print("=" * 80)
print("WORLD MODEL METRICS CALCULATION")
print("=" * 80)

results = []
world_dir = base / 'forecasts_WorldModel'

if not world_dir.exists():
    print(f"ERROR: World Model forecasts directory not found: {world_dir}")
    print("Please run RF_Market_WorldModel.py first.")
    exit(1)

for market_dir in sorted(world_dir.iterdir()):
    if not market_dir.is_dir():
        continue
    
    # Skip non-market directories
    if market_dir.name in ['rf_world_feature_importance.csv']:
        continue
    
    csvs = list((market_dir / 'rf').glob('*/test_pred.csv'))
    if not csvs:
        # Also check for consolidated file
        consolidated = market_dir / 'rf_pred.csv'
        if consolidated.exists():
            csvs = [consolidated]
        else:
            continue
    
    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    
    if len(df) == 0 or 'TARGET' not in df.columns:
        continue
    
    market = market_dir.name
    print(f"Processing {market}...", end=" ")
    
    # Load mvel1 from original data for value-weighting
    market_data = load_market_data(market)
    if market_data is not None:
        df['DATE'] = pd.to_datetime(df['DATE'])
        df['PERMNO'] = df['PERMNO'].astype(str)
        market_data['PERMNO'] = market_data['PERMNO'].astype(str)
        df = df.merge(market_data, on=['PERMNO', 'DATE'], how='left')
    
    # 1. R² OOS (per GKX: denominator is sum of squared returns, NOT demeaned)
    ss_res = ((df['TARGET'] - df['pred']) ** 2).sum()
    ss_tot = (df['TARGET'] ** 2).sum()
    r2_oos = 1 - ss_res / ss_tot
    
    # 2. MSE
    mse = ss_res / len(df)
    
    # 3. Rank Correlation (Spearman)
    rank_corr, _ = spearmanr(df['TARGET'], df['pred'])
    
    # 4 & 5. Sharpe Ratio and Decile Score Distance
    df['YearMonth'] = pd.to_datetime(df['DATE']).dt.to_period('M')
    
    monthly_sharpe_ew = []
    monthly_sharpe_vw = []
    decile_distances = []
    
    for ym, month_df in df.groupby('YearMonth'):
        if len(month_df) < 10:
            continue
        
        month_df = month_df.copy()
        try:
            month_df['pred_decile'] = pd.qcut(month_df['pred'], 10, labels=False, duplicates='drop') + 1
            month_df['actual_decile'] = pd.qcut(month_df['TARGET'], 10, labels=False, duplicates='drop') + 1
        except:
            continue
        
        long = month_df[month_df['pred_decile'] == month_df['pred_decile'].max()]
        short = month_df[month_df['pred_decile'] == month_df['pred_decile'].min()]
        
        if len(long) > 0 and len(short) > 0:
            # Equal-weighted returns
            long_ret_ew = long['TARGET'].mean()
            short_ret_ew = short['TARGET'].mean()
            monthly_sharpe_ew.append(long_ret_ew - short_ret_ew)
            
            # Value-weighted returns (using mvel1 = log market cap)
            if 'mvel1' in month_df.columns and month_df['mvel1'].notna().any():
                long_valid = long[long['mvel1'].notna()]
                short_valid = short[short['mvel1'].notna()]
                
                if len(long_valid) > 0 and len(short_valid) > 0:
                    # Convert log market cap back to market cap for weights
                    long_weights = np.exp(long_valid['mvel1'])
                    short_weights = np.exp(short_valid['mvel1'])
                    
                    # Normalize weights
                    long_weights = long_weights / long_weights.sum()
                    short_weights = short_weights / short_weights.sum()
                    
                    long_ret_vw = (long_valid['TARGET'] * long_weights).sum()
                    short_ret_vw = (short_valid['TARGET'] * short_weights).sum()
                else:
                    long_ret_vw = long_ret_ew
                    short_ret_vw = short_ret_ew
            else:
                long_ret_vw = long_ret_ew
                short_ret_vw = short_ret_ew
            
            monthly_sharpe_vw.append(long_ret_vw - short_ret_vw)
            
            # Decile Score Distance
            long_actual_decile = long['actual_decile'].mean()
            short_actual_decile = short['actual_decile'].mean()
            decile_distances.append(long_actual_decile - short_actual_decile)
    
    # Annualized Sharpe Ratio (multiply by sqrt(12))
    if monthly_sharpe_ew and np.std(monthly_sharpe_ew) > 0:
        sharpe_ew = (np.mean(monthly_sharpe_ew) / np.std(monthly_sharpe_ew)) * np.sqrt(12)
    else:
        sharpe_ew = np.nan
    
    if monthly_sharpe_vw and np.std(monthly_sharpe_vw) > 0:
        sharpe_vw = (np.mean(monthly_sharpe_vw) / np.std(monthly_sharpe_vw)) * np.sqrt(12)
    else:
        sharpe_vw = np.nan
    
    decile_dist = np.mean(decile_distances) if decile_distances else np.nan
    
    # Get test period
    min_date = pd.to_datetime(df['DATE']).min()
    max_date = pd.to_datetime(df['DATE']).max()
    test_period = f"{min_date.year}-{max_date.year}"
    
    results.append({
        'Market': market,
        'R2_OOS': round(r2_oos, 4),
        'MSE': round(mse, 6),
        'Rank_Corr': round(rank_corr, 4),
        'Sharpe_EW': round(sharpe_ew, 2) if not np.isnan(sharpe_ew) else np.nan,
        'Sharpe_VW': round(sharpe_vw, 2) if not np.isnan(sharpe_vw) else np.nan,
        'Decile_Dist': round(decile_dist, 2) if not np.isnan(decile_dist) else np.nan,
        'N': len(df),
        'Test_Period': test_period
    })
    
    print("done")

# =============================================================================
# SAVE RESULTS
# =============================================================================

summary = pd.DataFrame(results).sort_values('Market')
print("\n" + "=" * 80)
print("WORLD MODEL RESULTS SUMMARY")
print("=" * 80)
print(summary.to_string(index=False))

summary.to_csv(base / 'rf_WorldModel_results_summary.csv', index=False)
print(f"\nSaved to: {base / 'rf_WorldModel_results_summary.csv'}")

# Feature importance
importance_df = get_world_feature_importance()
if importance_df is not None:
    print("\n" + "=" * 80)
    print("TOP 10 FEATURES (WORLD MODEL)")
    print("=" * 80)
    print(importance_df.head(10).to_string(index=False))
    
    # Country dummy importance
    dummy_features = importance_df[importance_df['feature'].str.startswith('dummy_')]
    if len(dummy_features) > 0:
        print("\n" + "=" * 80)
        print("COUNTRY DUMMY IMPORTANCE (TOP 10)")
        print("=" * 80)
        print(dummy_features.head(10).to_string(index=False))

# =============================================================================
# COMPARISON: LOCAL vs US MODEL vs WORLD MODEL
# =============================================================================

print("\n" + "=" * 80)
print("LOADING COMPARISON DATA")
print("=" * 80)

local_results = None
us_model_results = None

local_path = base / 'rf_results_full_summary.csv'
us_path = base / 'rf_USmodel_results_summary.csv'

if local_path.exists():
    local_results = pd.read_csv(local_path)
    print(f"Loaded local model results: {len(local_results)} markets")

if us_path.exists():
    us_model_results = pd.read_csv(us_path)
    print(f"Loaded US model results: {len(us_model_results)} markets")

if local_results is not None and us_model_results is not None:
    # Merge all three result sets
    comparison = summary[['Market', 'R2_OOS', 'Sharpe_EW', 'Sharpe_VW', 'Rank_Corr', 'Decile_Dist']].copy()
    comparison.columns = ['Market', 'World_R2', 'World_SR_EW', 'World_SR_VW', 'World_RankCorr', 'World_Decile']
    
    # Add local results
    local_cols = local_results[['Market', 'R2_OOS', 'Sharpe_EW', 'Sharpe_VW', 'Rank_Corr', 'Decile_Dist']].copy()
    local_cols.columns = ['Market', 'Local_R2', 'Local_SR_EW', 'Local_SR_VW', 'Local_RankCorr', 'Local_Decile']
    comparison = comparison.merge(local_cols, on='Market', how='outer')
    
    # Add US model results
    us_cols = us_model_results[['Market', 'R2_OOS', 'Sharpe_EW', 'Sharpe_VW', 'Rank_Corr', 'Decile_Dist']].copy()
    us_cols.columns = ['Market', 'US_R2', 'US_SR_EW', 'US_SR_VW', 'US_RankCorr', 'US_Decile']
    comparison = comparison.merge(us_cols, on='Market', how='outer')
    
    # Calculate differences
    comparison['World_vs_Local_SR_EW'] = comparison['World_SR_EW'] - comparison['Local_SR_EW']
    comparison['World_vs_US_SR_EW'] = comparison['World_SR_EW'] - comparison['US_SR_EW']
    comparison['World_vs_Local_SR_VW'] = comparison['World_SR_VW'] - comparison['Local_SR_VW']
    comparison['World_vs_US_SR_VW'] = comparison['World_SR_VW'] - comparison['US_SR_VW']
    
    # Save comparison
    comparison.to_csv(base / 'rf_local_vs_USmodel_vs_WorldModel_comparison.csv', index=False)
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON: SHARPE RATIOS")
    print("=" * 80)
    
    display_cols = ['Market', 'Local_SR_EW', 'US_SR_EW', 'World_SR_EW', 
                    'Local_SR_VW', 'US_SR_VW', 'World_SR_VW']
    print(comparison[display_cols].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Summary statistics
    print("\nEqual-Weighted Sharpe Ratio:")
    print(f"  Local Model Average:  {comparison['Local_SR_EW'].mean():.2f}")
    print(f"  US Model Average:     {comparison['US_SR_EW'].mean():.2f}")
    print(f"  World Model Average:  {comparison['World_SR_EW'].mean():.2f}")
    
    world_beats_local_ew = (comparison['World_SR_EW'] > comparison['Local_SR_EW']).sum()
    world_beats_us_ew = (comparison['World_SR_EW'] > comparison['US_SR_EW']).sum()
    total_markets = len(comparison.dropna(subset=['World_SR_EW', 'Local_SR_EW', 'US_SR_EW']))
    
    print(f"\n  World beats Local: {world_beats_local_ew}/{total_markets} markets ({100*world_beats_local_ew/total_markets:.1f}%)")
    print(f"  World beats US:    {world_beats_us_ew}/{total_markets} markets ({100*world_beats_us_ew/total_markets:.1f}%)")
    
    print("\nValue-Weighted Sharpe Ratio:")
    print(f"  Local Model Average:  {comparison['Local_SR_VW'].mean():.2f}")
    print(f"  US Model Average:     {comparison['US_SR_VW'].mean():.2f}")
    print(f"  World Model Average:  {comparison['World_SR_VW'].mean():.2f}")
    
    world_beats_local_vw = (comparison['World_SR_VW'] > comparison['Local_SR_VW']).sum()
    world_beats_us_vw = (comparison['World_SR_VW'] > comparison['US_SR_VW']).sum()
    
    print(f"\n  World beats Local: {world_beats_local_vw}/{total_markets} markets ({100*world_beats_local_vw/total_markets:.1f}%)")
    print(f"  World beats US:    {world_beats_us_vw}/{total_markets} markets ({100*world_beats_us_vw/total_markets:.1f}%)")
    
    print(f"\nComparison saved to: {base / 'rf_local_vs_USmodel_vs_WorldModel_comparison.csv'}")

print("\n" + "=" * 80)
print("COMPLETED!")
print("=" * 80)