"""
Calculate all paper metrics: R² OOS, Sharpe Ratio, Rank Correlation, Decile Score Distance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

base = Path(r'C:\Users\matle\Desktop\VSCode\factor_model_project')

results = []

for market_dir in (base / 'forecasts').iterdir():
    if not market_dir.is_dir():
        continue
    
    # Load predictions
    csvs = list((market_dir / 'rf').glob('*/test_pred.csv'))
    if not csvs:
        continue
    
    df = pd.concat([pd.read_csv(f) for f in csvs])
    
    if len(df) == 0 or 'TARGET' not in df.columns:
        continue
    
    market = market_dir.name
    
    # 1. R² OOS
    ss_res = ((df['TARGET'] - df['pred']) ** 2).sum()
    ss_tot = (df['TARGET'] ** 2).sum()  # Paper uses non-demeaned denominator
    r2_oos = 1 - ss_res / ss_tot
    
    # 2. Rank Correlation (Spearman)
    rank_corr, _ = spearmanr(df['TARGET'], df['pred'])
    
    # 3 & 4. Sharpe Ratio and Decile Score Distance (need monthly aggregation)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['YearMonth'] = df['DATE'].dt.to_period('M')
    
    monthly_sharpe_ew = []
    monthly_sharpe_vw = []
    decile_distances = []
    
    for ym, month_df in df.groupby('YearMonth'):
        if len(month_df) < 10:
            continue
        
        # Assign deciles based on predicted returns
        month_df = month_df.copy()
        month_df['pred_decile'] = pd.qcut(month_df['pred'], 10, labels=False, duplicates='drop') + 1
        month_df['actual_decile'] = pd.qcut(month_df['TARGET'], 10, labels=False, duplicates='drop') + 1
        
        # Long-short portfolio returns (decile 10 - decile 1)
        long = month_df[month_df['pred_decile'] == month_df['pred_decile'].max()]
        short = month_df[month_df['pred_decile'] == month_df['pred_decile'].min()]
        
        if len(long) > 0 and len(short) > 0:
            # Equal-weighted
            long_ret_ew = long['TARGET'].mean()
            short_ret_ew = short['TARGET'].mean()
            monthly_sharpe_ew.append(long_ret_ew - short_ret_ew)
            
            # Value-weighted (use mvel1 if available, else equal-weight)
            if 'mvel1' in month_df.columns:
                long_ret_vw = np.average(long['TARGET'], weights=np.exp(long['mvel1']))
                short_ret_vw = np.average(short['TARGET'], weights=np.exp(short['mvel1']))
            else:
                long_ret_vw = long_ret_ew
                short_ret_vw = short_ret_ew
            monthly_sharpe_vw.append(long_ret_vw - short_ret_vw)
            
            # Decile Score Distance
            long_actual_decile = long['actual_decile'].mean()
            short_actual_decile = short['actual_decile'].mean()
            decile_distances.append(long_actual_decile - short_actual_decile)
    
    # Calculate annualized Sharpe Ratios
    if monthly_sharpe_ew:
        sharpe_ew = (np.mean(monthly_sharpe_ew) / np.std(monthly_sharpe_ew)) * np.sqrt(12)
        sharpe_vw = (np.mean(monthly_sharpe_vw) / np.std(monthly_sharpe_vw)) * np.sqrt(12)
        decile_dist = np.mean(decile_distances)
    else:
        sharpe_ew = sharpe_vw = decile_dist = np.nan
    
    results.append({
        'Market': market,
        'R2_OOS': round(r2_oos, 4),
        'Rank_Corr': round(rank_corr, 4),
        'Sharpe_EW': round(sharpe_ew, 2),
        'Sharpe_VW': round(sharpe_vw, 2),
        'Decile_Dist': round(decile_dist, 2),
        'N': len(df)
    })

# Create summary
summary = pd.DataFrame(results).sort_values('Market')
print(summary.to_string(index=False))

# Save
summary.to_csv(base / 'rf_results_full_summary.csv', index=False)
print(f"\nSaved to: {base / 'rf_results_full_summary.csv'}")
