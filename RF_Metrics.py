"""
Calculate all paper metrics: R² OOS, MSE, Sharpe Ratio, Rank Correlation, Decile Score Distance,
Relative Importance of Predictors, and CKA Similarity Index
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from joblib import load
import sys

base = Path(r'C:\Users\matle\Desktop\VSCode\factor_model_project')

# Add project path for imports
sys.path.insert(0, str(base))

# Feature names (36 features used in the paper)
FEATURE_NAMES = [
    'absacc', 'acc', 'age', 'agr', 'bm', 'bm_ia', 'cashdebt', 'cashpr',
    'cfp', 'cfp_ia', 'chatoia', 'chcsho', 'chempia', 'chinv', 'chpmia',
    'convind', 'currat', 'depr', 'divi', 'divo', 'dy', 'egr', 'ep', 'gma',
    'grcapx', 'grltnoa', 'herf', 'hire', 'invest', 'lev', 'lgr', 'mvel1',
    'operprof', 'orgcap', 'pchcurrat', 'rd'
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_feature_importance(market):
    """Load feature importance from saved CSV or calculate from models."""
    # Try loading from saved file first
    importance_file = base / 'forecasts' / market / 'rf_feature_importance.csv'
    if importance_file.exists():
        return pd.read_csv(importance_file)
    
    # Otherwise calculate from models
    model_dir = base / 'model_parameters' / market / 'rf'
    if not model_dir.exists():
        return None
    
    model_files = list(model_dir.glob('year*.joblib'))
    if not model_files:
        return None
    
    # Get feature count from first model
    first_model = load(model_files[0])
    n_features = first_model.n_features_in_
    
    # Only use models with same number of features
    all_importances = []
    for mf in model_files:
        model = load(mf)
        if model.n_features_in_ == n_features:
            all_importances.append(model.feature_importances_)
    
    if not all_importances:
        return None
    
    avg_importance = np.mean(all_importances, axis=0)
    
    # Use feature names (truncate if fewer features)
    feature_names = FEATURE_NAMES[:n_features] if n_features <= len(FEATURE_NAMES) else [f'feature_{i}' for i in range(n_features)]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)
    importance_df['importance_pct'] = (importance_df['importance'] / importance_df['importance'].sum() * 100).round(2)
    
    return importance_df


def calculate_cka_similarity(market, reference_market='USA'):
    """
    Calculate CKA (Centered Kernel Alignment) similarity between market and reference.
    Higher CKA = more similar learned relationships.
    """
    if market == reference_market:
        return 1.0
    
    market_model_dir = base / 'model_parameters' / market / 'rf'
    ref_model_dir = base / 'model_parameters' / reference_market / 'rf'
    
    if not market_model_dir.exists() or not ref_model_dir.exists():
        return np.nan
    
    # Get common years
    market_years = {int(f.stem.replace('year', '')) for f in market_model_dir.glob('year*.joblib')}
    ref_years = {int(f.stem.replace('year', '')) for f in ref_model_dir.glob('year*.joblib')}
    common_years = sorted(market_years & ref_years)
    
    if not common_years:
        return np.nan
    
    market_importances = []
    ref_importances = []
    
    for year in common_years:
        market_model_path = market_model_dir / f'year{year}.joblib'
        ref_model_path = ref_model_dir / f'year{year}.joblib'
        
        if market_model_path.exists() and ref_model_path.exists():
            market_model = load(market_model_path)
            ref_model = load(ref_model_path)
            
            # Only include if both have same number of features
            if market_model.n_features_in_ == ref_model.n_features_in_:
                market_importances.append(market_model.feature_importances_)
                ref_importances.append(ref_model.feature_importances_)
    
    if len(market_importances) < 2:
        return np.nan
    
    # Stack importances
    X = np.array(market_importances)
    Y = np.array(ref_importances)
    
    # Linear CKA calculation
    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y
    
    numerator = np.linalg.norm(YtX, 'fro') ** 2
    denominator = np.linalg.norm(XtX, 'fro') * np.linalg.norm(YtY, 'fro')
    
    if denominator == 0:
        return np.nan
    
    cka = numerator / denominator
    
    return cka


# =============================================================================
# MAIN METRICS CALCULATION
# =============================================================================

results = []
importance_results = []

for market_dir in sorted((base / 'forecasts').iterdir()):
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
    print(f"Processing {market}...", end=" ")
    
    # 1. R² OOS (paper uses non-demeaned denominator)
    ss_res = ((df['TARGET'] - df['pred']) ** 2).sum()
    ss_tot = (df['TARGET'] ** 2).sum()
    r2_oos = 1 - ss_res / ss_tot
    
    # 2. MSE
    mse = ss_res / len(df)
    
    # 3. Rank Correlation (Spearman)
    rank_corr, _ = spearmanr(df['TARGET'], df['pred'])
    
    # 4 & 5. Sharpe Ratio and Decile Score Distance
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['YearMonth'] = df['DATE'].dt.to_period('M')
    
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
    
    # 6. CKA Similarity with USA
    cka_similarity = calculate_cka_similarity(market, 'USA')
    
    # 7. Feature Importance (top 5)
    importance_df = get_feature_importance(market)
    if importance_df is not None and len(importance_df) > 0:
        top_5_features = importance_df.head(5)['feature'].tolist()
        
        # Save full importance for this market
        for _, row in importance_df.iterrows():
            importance_results.append({
                'Market': market,
                'Feature': row['feature'],
                'Importance': row['importance'],
                'Importance_Pct': row['importance_pct']
            })
    else:
        top_5_features = []
    
    results.append({
        'Market': market,
        'R2_OOS': round(r2_oos, 4),
        'MSE': round(mse, 6),
        'Rank_Corr': round(rank_corr, 4),
        'Sharpe_EW': round(sharpe_ew, 2) if not np.isnan(sharpe_ew) else np.nan,
        'Sharpe_VW': round(sharpe_vw, 2) if not np.isnan(sharpe_vw) else np.nan,
        'Decile_Dist': round(decile_dist, 2) if not np.isnan(decile_dist) else np.nan,
        'CKA_vs_USA': round(cka_similarity, 4) if not np.isnan(cka_similarity) else np.nan,
        'N': len(df),
        'Top_Features': ', '.join(top_5_features[:3]) if top_5_features else ''
    })
    
    print("done")

# =============================================================================
# CREATE AND SAVE SUMMARIES
# =============================================================================

# Main results summary
summary = pd.DataFrame(results).sort_values('Market')
print("\n" + "=" * 80)
print("FULL RESULTS SUMMARY")
print("=" * 80)
print(summary.to_string(index=False))

summary.to_csv(base / 'rf_results_full_summary.csv', index=False)
print(f"\nSaved to: {base / 'rf_results_full_summary.csv'}")

# Feature importance summary (all markets, all features)
if importance_results:
    importance_summary = pd.DataFrame(importance_results)
    importance_summary.to_csv(base / 'rf_feature_importance_all_markets.csv', index=False)
    print(f"Saved to: {base / 'rf_feature_importance_all_markets.csv'}")
    
    # Top features across all markets
    print("\n" + "=" * 80)
    print("TOP 10 FEATURES BY AVERAGE IMPORTANCE ACROSS MARKETS")
    print("=" * 80)
    avg_importance = importance_summary.groupby('Feature')['Importance_Pct'].mean().sort_values(ascending=False)
    print(avg_importance.head(10).to_string())

# CKA Summary
print("\n" + "=" * 80)
print("CKA SIMILARITY WITH USA (higher = more similar)")
print("=" * 80)
cka_df = summary[['Market', 'CKA_vs_USA']].dropna().sort_values('CKA_vs_USA', ascending=False)
print(cka_df.to_string(index=False))

print(f"\nCompleted!")