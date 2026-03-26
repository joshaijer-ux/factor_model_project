"""
RF_Market_USmodel.py
Apply US-trained Random Forest models to International Markets.

This is the "Alpha Goes Everywhere" test from Choi, Jiang, Zhang (2025).
The US model is trained on US data, then applied to predict returns in other markets
WITHOUT retraining. This tests whether US return-characteristic relationships
transfer to international markets.

Based on ISLP textbook Chapter 8 and paper methodology.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, r'C:\Users\matle\Desktop\VSCode\factor_model_project')

from joblib import load
from scipy.stats import spearmanr

from RF_Load_Data import load_data, in_output, load_markets

# =============================================================================
# CONFIGURATION - RELATIVE PATHS
# =============================================================================

base = Path(__file__).parent if '__file__' in dir() else Path.cwd()
YOUR_PATH = str(base)

# Path to US trained models
US_MODEL_PATH = base / 'model_parameters' / 'USA' / 'rf'


# =============================================================================
# APPLY US MODEL TO INTERNATIONAL MARKET
# =============================================================================

def apply_us_model_to_market(market):
    """
    Apply US-trained RF models to an international market.
    
    For each year in the international market's test period:
    1. Load the US model trained for that year
    2. Apply it to the international market's data for that year
    3. Generate predictions
    """
    print('\n' + '=' * 60)
    print(f"APPLYING US MODEL TO: {market}")
    print('=' * 60)
    
    if market == 'USA':
        print("Skipping USA (this is the source market)")
        return None
    
    # Load international market data
    final_data, start_year, train_split, valid_split, end_year, batch_size = load_data(YOUR_PATH, market)
    
    if len(final_data) == 0:
        print(f"No data for {market}, skipping...")
        return None
    
    # Get feature names
    sample_X, _ = in_output(final_data.head())
    feature_names = list(sample_X.columns)
    print(f"Features: {len(feature_names)}")
    
    # Get available US model years
    us_model_files = list(US_MODEL_PATH.glob('year*.joblib'))
    if not us_model_files:
        print("ERROR: No US models found. Run RF_Market.py for USA first.")
        return None
    
    us_model_years = sorted([int(f.stem.replace('year', '')) for f in us_model_files])
    print(f"US models available: {min(us_model_years)}-{max(us_model_years)}")
    
    # Determine test years for international market
    # Use years where both: (1) intl market has test data, (2) US model exists
    intl_test_years = range(valid_split + 1, end_year + 1)
    common_years = [y for y in intl_test_years if y in us_model_years]
    
    if not common_years:
        print(f"No overlapping years between US models and {market} test period")
        return None
    
    print(f"Test years: {min(common_years)}-{max(common_years)}")
    
    # Apply US model for each year
    start_time = time.time()
    all_predictions = []
    
    for year in common_years:
        print(f"\n  Year {year}: ", end="", flush=True)
        
        # Load US model for this year
        model_path = US_MODEL_PATH / f'year{year}.joblib'
        if not model_path.exists():
            print(f"US model not found, skipping", end="")
            continue
        
        us_model = load(model_path)
        
        # Get international market data for this year
        year_data = final_data[final_data['DATE'].dt.year == year].copy()
        
        if len(year_data) == 0:
            print(f"No data, skipping", end="")
            continue
        
        # Separate features and target
        X, y = in_output(year_data)
        X = np.asarray(X)
        
        # Generate predictions using US model
        pred = us_model.predict(X)
        
        # Create prediction dataframe
        year_data = year_data.reset_index(drop=True)
        pred_df = pd.concat([
            year_data[['PERMNO', 'DATE', 'TARGET']],
            pd.DataFrame(pred, columns=['pred'])
        ], axis=1)
        
        all_predictions.append(pred_df)
        print(f"{len(year_data):,} obs", end="")
    
    total_time = time.time() - start_time
    
    if not all_predictions:
        print(f"\nNo predictions generated for {market}")
        return None
    
    # Combine all predictions
    forecast = pd.concat(all_predictions, ignore_index=True)
    
    # ==========================================================================
    # CALCULATE ALL METRICS
    # ==========================================================================
    
    # 1. R² OOS (using non-demeaned denominator as in paper)
    ss_res = ((forecast['TARGET'] - forecast['pred']) ** 2).sum()
    ss_tot = (forecast['TARGET'] ** 2).sum()
    r2_oos = 1 - ss_res / ss_tot
    
    # 2. R² OOS (traditional, demeaned)
    ss_tot_demeaned = ((forecast['TARGET'] - forecast['TARGET'].mean()) ** 2).sum()
    r2_oos_demeaned = 1 - ss_res / ss_tot_demeaned
    
    # 3. MSE
    mse = ss_res / len(forecast)
    
    # 4. Rank Correlation
    rank_corr, _ = spearmanr(forecast['TARGET'], forecast['pred'])
    
    # 5 & 6. Sharpe Ratio and Decile Score Distance
    forecast['DATE'] = pd.to_datetime(forecast['DATE'])
    forecast['YearMonth'] = forecast['DATE'].dt.to_period('M')
    
    monthly_sharpe_ew = []
    monthly_sharpe_vw = []
    decile_distances = []
    
    for ym, month_df in forecast.groupby('YearMonth'):
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
            long_ret_ew = long['TARGET'].mean()
            short_ret_ew = short['TARGET'].mean()
            monthly_sharpe_ew.append(long_ret_ew - short_ret_ew)
            
            if 'mvel1' in month_df.columns:
                try:
                    long_ret_vw = np.average(long['TARGET'], weights=np.exp(long['mvel1']))
                    short_ret_vw = np.average(short['TARGET'], weights=np.exp(short['mvel1']))
                except:
                    long_ret_vw = long_ret_ew
                    short_ret_vw = short_ret_ew
            else:
                long_ret_vw = long_ret_ew
                short_ret_vw = short_ret_ew
            monthly_sharpe_vw.append(long_ret_vw - short_ret_vw)
            
            long_actual_decile = long['actual_decile'].mean()
            short_actual_decile = short['actual_decile'].mean()
            decile_distances.append(long_actual_decile - short_actual_decile)
    
    if monthly_sharpe_ew and np.std(monthly_sharpe_ew) > 0:
        sharpe_ew = (np.mean(monthly_sharpe_ew) / np.std(monthly_sharpe_ew)) * np.sqrt(12)
        sharpe_vw = (np.mean(monthly_sharpe_vw) / np.std(monthly_sharpe_vw)) * np.sqrt(12) if np.std(monthly_sharpe_vw) > 0 else np.nan
        decile_dist = np.mean(decile_distances)
    else:
        sharpe_ew = sharpe_vw = decile_dist = np.nan
    
    print(f"\n\n{'─' * 40}")
    print(f"RESULTS: {market} (US Model)")
    print(f"{'─' * 40}")
    print(f"R² OOS (paper): {r2_oos:.4f}")
    print(f"R² OOS (trad):  {r2_oos_demeaned:.4f}")
    print(f"MSE:            {mse:.6f}")
    print(f"Rank Corr:      {rank_corr:.4f}")
    print(f"Sharpe EW:      {sharpe_ew:.2f}" if not np.isnan(sharpe_ew) else "Sharpe EW:      N/A")
    print(f"Sharpe VW:      {sharpe_vw:.2f}" if not np.isnan(sharpe_vw) else "Sharpe VW:      N/A")
    print(f"Decile Dist:    {decile_dist:.2f}" if not np.isnan(decile_dist) else "Decile Dist:    N/A")
    print(f"N test obs:     {len(forecast):,}")
    print(f"Test period:    {min(common_years)}-{max(common_years)}")
    print(f"Time:           {total_time:.1f}s")
    
    # Save predictions
    save_dir = base / 'forecasts_USmodel' / market
    os.makedirs(save_dir, exist_ok=True)
    forecast.to_csv(save_dir / 'rf_pred.csv', index=False)
    
    return {
        'market': market,
        'r2_oos': r2_oos,
        'r2_oos_demeaned': r2_oos_demeaned,
        'mse': mse,
        'rank_corr': rank_corr,
        'sharpe_ew': sharpe_ew,
        'sharpe_vw': sharpe_vw,
        'decile_dist': decile_dist,
        'n_obs': len(forecast),
        'test_start': min(common_years),
        'test_end': max(common_years),
        'time_seconds': total_time
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("ALPHA GO EVERYWHERE - US MODEL TRANSFER")
    print("Applying US-trained Random Forest to International Markets")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project path: {YOUR_PATH}")
    print()
    
    # Check US models exist
    if not US_MODEL_PATH.exists():
        print("ERROR: US model folder not found.")
        print(f"Expected: {US_MODEL_PATH}")
        print("Run RF_Market.py with USA first.")
        exit(1)
    
    us_models = list(US_MODEL_PATH.glob('year*.joblib'))
    print(f"Found {len(us_models)} US models")
    print()
    
    # Get all markets
    markets = load_markets(YOUR_PATH)
    
    if not markets:
        print("ERROR: No markets found.")
        exit(1)
    
    # Remove USA from list (it's the source, not target)
    intl_markets = [m for m in markets if m != 'USA']
    print(f"International markets to test: {len(intl_markets)}")
    print()
    
    # Apply US model to each international market
    all_results = []
    
    for market in intl_markets:
        try:
            result = apply_us_model_to_market(market)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"\nERROR with {market}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if all_results:
        print("\n" + "=" * 60)
        print("SUMMARY: US MODEL APPLIED TO INTERNATIONAL MARKETS")
        print("=" * 60)
        
        summary_df = pd.DataFrame([{
            'Market': r['market'],
            'R2_OOS': round(r['r2_oos'], 4),
            'MSE': round(r['mse'], 6),
            'Rank_Corr': round(r['rank_corr'], 4),
            'Sharpe_EW': round(r['sharpe_ew'], 2) if not np.isnan(r['sharpe_ew']) else np.nan,
            'Sharpe_VW': round(r['sharpe_vw'], 2) if not np.isnan(r['sharpe_vw']) else np.nan,
            'Decile_Dist': round(r['decile_dist'], 2) if not np.isnan(r['decile_dist']) else np.nan,
            'N': r['n_obs'],
            'Test_Period': f"{r['test_start']}-{r['test_end']}"
        } for r in all_results])
        
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_df.to_csv(base / 'rf_USmodel_results_summary.csv', index=False)
        print(f"\nSummary saved: {base / 'rf_USmodel_results_summary.csv'}")
        
        # Compare with local models (if available)
        local_results_path = base / 'rf_results_full_summary.csv'
        if local_results_path.exists():
            print("\n" + "=" * 60)
            print("COMPARISON: LOCAL MODEL vs US MODEL")
            print("=" * 60)
            
            local_df = pd.read_csv(local_results_path)
            
            comparison = []
            for r in all_results:
                local_row = local_df[local_df['Market'] == r['market']]
                if len(local_row) > 0:
                    local_r2 = local_row['R2_OOS'].values[0]
                    local_sharpe = local_row['Sharpe_EW'].values[0] if 'Sharpe_EW' in local_row.columns else np.nan
                    local_rank = local_row['Rank_Corr'].values[0] if 'Rank_Corr' in local_row.columns else np.nan
                    
                    us_r2 = r['r2_oos']
                    us_sharpe = r['sharpe_ew']
                    us_rank = r['rank_corr']
                    
                    comparison.append({
                        'Market': r['market'],
                        'Local_R2': round(local_r2, 4),
                        'US_R2': round(us_r2, 4),
                        'R2_Diff': round(local_r2 - us_r2, 4),
                        'Local_Sharpe': round(local_sharpe, 2) if not np.isnan(local_sharpe) else np.nan,
                        'US_Sharpe': round(us_sharpe, 2) if not np.isnan(us_sharpe) else np.nan,
                        'Local_Rank': round(local_rank, 4) if not np.isnan(local_rank) else np.nan,
                        'US_Rank': round(us_rank, 4),
                        'Local_Better_R2': 'Yes' if local_r2 > us_r2 else 'No'
                    })
            
            if comparison:
                comp_df = pd.DataFrame(comparison)
                print(comp_df.to_string(index=False))
                
                local_wins_r2 = sum(1 for c in comparison if c['Local_Better_R2'] == 'Yes')
                print(f"\nLocal model better (R²) in {local_wins_r2}/{len(comparison)} markets")
                
                comp_df.to_csv(base / 'rf_local_vs_USmodel_comparison.csv', index=False)
                print(f"Comparison saved: {base / 'rf_local_vs_USmodel_comparison.csv'}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")