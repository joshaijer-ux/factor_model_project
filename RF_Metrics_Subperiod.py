"""
RF_Metrics_SubPeriods.py
Calculate metrics for sub-periods using existing forecast files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import sys

# =============================================================================
# CONFIGURATION - UPDATE THIS PATH TO YOUR PROJECT FOLDER
# =============================================================================

base = Path(r'C:\Users\matle\Desktop\VSCode\factor_model_project')
sys.path.insert(0, str(base))

# Paper's test period cutoff
PAPER_CUTOFF = '2017-12-31'


def load_market_data(market):
    """Load original market data to get mvel1 for value-weighting."""
    data_path = base / 'normalized' / f'{market}_ranked.parquet'
    if data_path.exists():
        df = pd.read_parquet(data_path)
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df[['PERMNO', 'DATE', 'mvel1']].copy()
    return None


def calculate_metrics(df):
    """Calculate all metrics for a given dataframe."""
    if len(df) == 0 or 'TARGET' not in df.columns:
        return None
    
    # 1. R² OOS
    ss_res = ((df['TARGET'] - df['pred']) ** 2).sum()
    ss_tot = (df['TARGET'] ** 2).sum()
    r2_oos = 1 - ss_res / ss_tot
    
    # 2. MSE
    mse = ss_res / len(df)
    
    # 3. Rank Correlation
    rank_corr, _ = spearmanr(df['TARGET'], df['pred'])
    
    # 4 & 5. Sharpe Ratio and Decile Score Distance
    df = df.copy()
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
            
            # Value-weighted returns
            if 'mvel1' in month_df.columns and month_df['mvel1'].notna().any():
                long_valid = long[long['mvel1'].notna()]
                short_valid = short[short['mvel1'].notna()]
                
                if len(long_valid) > 0 and len(short_valid) > 0:
                    long_weights = np.exp(long_valid['mvel1'])
                    short_weights = np.exp(short_valid['mvel1'])
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
    
    if monthly_sharpe_ew and np.std(monthly_sharpe_ew) > 0:
        sharpe_ew = (np.mean(monthly_sharpe_ew) / np.std(monthly_sharpe_ew)) * np.sqrt(12)
    else:
        sharpe_ew = np.nan
    
    if monthly_sharpe_vw and np.std(monthly_sharpe_vw) > 0:
        sharpe_vw = (np.mean(monthly_sharpe_vw) / np.std(monthly_sharpe_vw)) * np.sqrt(12)
    else:
        sharpe_vw = np.nan
    
    decile_dist = np.mean(decile_distances) if decile_distances else np.nan
    
    # Get date range
    min_date = pd.to_datetime(df['DATE']).min()
    max_date = pd.to_datetime(df['DATE']).max()
    
    return {
        'R2_OOS': round(r2_oos, 4),
        'MSE': round(mse, 6),
        'Rank_Corr': round(rank_corr, 4),
        'Sharpe_EW': round(sharpe_ew, 2) if not np.isnan(sharpe_ew) else np.nan,
        'Sharpe_VW': round(sharpe_vw, 2) if not np.isnan(sharpe_vw) else np.nan,
        'Decile_Dist': round(decile_dist, 2) if not np.isnan(decile_dist) else np.nan,
        'N': len(df),
        'Start': min_date.strftime('%Y-%m'),
        'End': max_date.strftime('%Y-%m')
    }


def process_model_forecasts(forecasts_dir, model_name):
    """Process forecasts and return results for all periods."""
    
    print(f"\n  Processing {model_name}...", end=" ")
    
    results = []
    
    for market_dir in sorted(forecasts_dir.iterdir()):
        if not market_dir.is_dir():
            continue
        
        # Find prediction files
        if (market_dir / 'rf').exists():
            csvs = list((market_dir / 'rf').glob('*/test_pred.csv'))
        else:
            csvs = list(market_dir.glob('rf_pred.csv'))
            if not csvs:
                csvs = list(market_dir.glob('*_pred.csv'))
        
        if not csvs:
            continue
        
        df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
        
        if len(df) == 0 or 'TARGET' not in df.columns:
            continue
        
        market = market_dir.name
        
        # Parse DATE
        df['DATE'] = pd.to_datetime(df['DATE'])

        # FIX: Drop mvel1 and YearMonth if already present in forecast file
        # (US Transfer forecasts include these columns from RF_Market_USmodel.py).
        # Without this, merging below produces mvel1_x / mvel1_y, and the
        # 'mvel1' column check in calculate_metrics() silently fails → VW = EW.
        for col in ['mvel1', 'YearMonth']:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Load mvel1 for value-weighting
        market_data = load_market_data(market)
        if market_data is not None:
            df['PERMNO'] = df['PERMNO'].astype(str)
            market_data['PERMNO'] = market_data['PERMNO'].astype(str)
            df = df.merge(market_data, on=['PERMNO', 'DATE'], how='left')

            # Warn if coverage is low (indicates a merge problem)
            coverage = df['mvel1'].notna().mean()
            if coverage < 0.5:
                print(f"\n  WARNING: {market} ({model_name}) mvel1 coverage = {coverage:.1%} — check PERMNO format")
        
        # Split by period
        df_pre2018 = df[df['DATE'] <= PAPER_CUTOFF].copy()
        df_post2017 = df[df['DATE'] > PAPER_CUTOFF].copy()
        
        # Calculate metrics for each period
        for period_name, period_df in [('Full', df), ('Pre-2018', df_pre2018), ('Post-2017', df_post2017)]:
            if len(period_df) > 0:
                metrics = calculate_metrics(period_df)
                if metrics:
                    metrics['Market'] = market
                    metrics['Model'] = model_name
                    metrics['Period'] = period_name
                    results.append(metrics)
    
    print(f"{len(results)} rows")
    return results


# =============================================================================
# MAIN
# =============================================================================

print("=" * 70)
print("RF METRICS BY SUB-PERIOD")
print(f"Paper cutoff: {PAPER_CUTOFF}")
print("=" * 70)

all_results = []

# Process each model type
model_configs = [
    (base / 'forecasts', 'Local'),
    (base / 'forecasts_USmodel', 'US_Transfer'),
    (base / 'forecasts_WorldModel', 'World'),
]

for forecasts_dir, model_name in model_configs:
    if forecasts_dir.exists():
        results = process_model_forecasts(forecasts_dir, model_name)
        all_results.extend(results)

# =============================================================================
# SAVE RESULTS
# =============================================================================

# Create main results dataframe
results_df = pd.DataFrame(all_results)

# Reorder columns
cols = ['Model', 'Market', 'Period', 'R2_OOS', 'Sharpe_EW', 'Sharpe_VW', 
        'Rank_Corr', 'Decile_Dist', 'N', 'Start', 'End']
cols = [c for c in cols if c in results_df.columns]
results_df = results_df[cols].sort_values(['Model', 'Market', 'Period'])

# Save main results file
results_df.to_csv(base / 'rf_subperiod_results.csv', index=False)
print(f"\nSaved: rf_subperiod_results.csv")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

summary = results_df.groupby(['Model', 'Period']).agg({
    'R2_OOS': 'mean',
    'Sharpe_EW': 'mean',
    'Sharpe_VW': 'mean',
    'Rank_Corr': 'mean',
    'Decile_Dist': 'mean',
    'Market': 'count'
}).round(4)
summary.columns = ['Avg_R2', 'Avg_SR_EW', 'Avg_SR_VW', 'Avg_RankCorr', 'Avg_Decile', 'N_Markets']
summary = summary.reset_index()

summary.to_csv(base / 'rf_subperiod_summary.csv', index=False)
print(f"Saved: rf_subperiod_summary.csv")

print("\n" + "=" * 70)
print("SUMMARY: AVERAGE METRICS BY MODEL AND PERIOD")
print("=" * 70)
print(summary.to_string(index=False))

print("\n" + "=" * 70)
print("COMPLETED!")
print("=" * 70)
