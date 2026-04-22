import numpy as np
import pandas as pd
import os
import time
import warnings
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from joblib import dump, load
import xgboost as xgb
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR        = Path(__file__).parent if '__file__' in dir() else Path.cwd()
NORMALIZED_PATH = BASE_DIR / 'normalized'
WORKING_DIR     = BASE_DIR / 'output'
MODEL_DIR       = WORKING_DIR / 'model_parameters'
FORECAST_DIR    = WORKING_DIR / 'forecasts'
US_MODEL_PATH   = MODEL_DIR / 'USA' / 'gbrt'

for d in [WORKING_DIR, MODEL_DIR, FORECAST_DIR]:
    os.makedirs(str(d), exist_ok=True)

print(f"Base dir:        {BASE_DIR}")
print(f"Normalized data: {NORMALIZED_PATH}")
print(f"Output:          {WORKING_DIR}")

# =============================================================================
# MARKET INFO
# =============================================================================
MARKET_INFO = {
    'USA': {'start': 1963, 'train': 1979, 'valid': 1989, 'end': 2024, 'batch_size': 10000},
    'JPN': {'start': 2008, 'train': 2010, 'valid': 2011, 'end': 2024, 'batch_size': 5000},
    'CHN': {'start': 1999, 'train': 2004, 'valid': 2007, 'end': 2024, 'batch_size': 3000},
    'IND': {'start': 2007, 'train': 2010, 'valid': 2012, 'end': 2024, 'batch_size': 3000},
    'KOR': {'start': 1997, 'train': 2003, 'valid': 2007, 'end': 2024, 'batch_size': 3000},
    'HKG': {'start': 1997, 'train': 2003, 'valid': 2007, 'end': 2024, 'batch_size': 2000},
    'TWN': {'start': 2007, 'train': 2010, 'valid': 2012, 'end': 2024, 'batch_size': 2000},
    'FRA': {'start': 1995, 'train': 2001, 'valid': 2005, 'end': 2024, 'batch_size': 2000},
    'GBR': {'start': 2005, 'train': 2008, 'valid': 2010, 'end': 2024, 'batch_size': 2000},
    'THA': {'start': 1997, 'train': 2003, 'valid': 2007, 'end': 2024, 'batch_size': 2000},
    'AUS': {'start': 2008, 'train': 2010, 'valid': 2011, 'end': 2024, 'batch_size': 2000},
    'SGP': {'start': 2007, 'train': 2010, 'valid': 2012, 'end': 2024, 'batch_size': 2000},
    'SWE': {'start': 2001, 'train': 2005, 'valid': 2008, 'end': 2024, 'batch_size': 2000},
    'ZAF': {'start': 1997, 'train': 2003, 'valid': 2007, 'end': 2024, 'batch_size': 2000},
    'POL': {'start': 2006, 'train': 2009, 'valid': 2011, 'end': 2024, 'batch_size': 2000},
    'ISR': {'start': 2005, 'train': 2008, 'valid': 2010, 'end': 2024, 'batch_size': 2000},
    'VNM': {'start': 2010, 'train': 2012, 'valid': 2013, 'end': 2024, 'batch_size': 2000},
    'ITA': {'start': 2001, 'train': 2005, 'valid': 2008, 'end': 2024, 'batch_size': 2000},
    'TUR': {'start': 2006, 'train': 2009, 'valid': 2011, 'end': 2024, 'batch_size': 2000},
    'CHE': {'start': 2002, 'train': 2006, 'valid': 2009, 'end': 2024, 'batch_size': 2000},
    'IDN': {'start': 2005, 'train': 2008, 'valid': 2010, 'end': 2024, 'batch_size': 2000},
    'GRC': {'start': 2006, 'train': 2009, 'valid': 2011, 'end': 2024, 'batch_size': 2000},
    'PHL': {'start': 2006, 'train': 2009, 'valid': 2011, 'end': 2024, 'batch_size': 2000},
    'NOR': {'start': 2007, 'train': 2010, 'valid': 2012, 'end': 2024, 'batch_size': 2000},
    'LKA': {'start': 2010, 'train': 2012, 'valid': 2013, 'end': 2024, 'batch_size': 2000},
    'DNK': {'start': 2007, 'train': 2010, 'valid': 2012, 'end': 2024, 'batch_size': 2000},
    'FIN': {'start': 2007, 'train': 2010, 'valid': 2012, 'end': 2024, 'batch_size': 2000},
    'SAU': {'start': 2010, 'train': 2012, 'valid': 2013, 'end': 2024, 'batch_size': 2000},
    'JOR': {'start': 2009, 'train': 2011, 'valid': 2012, 'end': 2024, 'batch_size': 2000},
    'EGY': {'start': 2010, 'train': 2012, 'valid': 2013, 'end': 2024, 'batch_size': 2000},
    'ESP': {'start': 2011, 'train': 2012, 'valid': 2013, 'end': 2024, 'batch_size': 2000},
    'KWT': {'start': 2012, 'train': 2013, 'valid': 2014, 'end': 2024, 'batch_size': 2000},
}

# =============================================================================
# HYPERPARAMETER GRID  (paper Appendix C)
# =============================================================================
PARAM_GRID = {
    'max_depth':     [1, 2],
    'learning_rate': [0.01, 0.1],
    'n_estimators':  [1000],
}
XGB_OBJECTIVE = 'reg:pseudohubererror'

# =============================================================================
# SHARED HELPERS
# =============================================================================

def _monthly_vw_spread(month_df):
    """Value-weighted long-minus-short monthly return using mvel1_raw as weight."""
    mdf = month_df.copy()
    for col in ('pred', 'TARGET', 'mvel1_raw'):
        mdf[col] = pd.to_numeric(mdf[col], errors='coerce')
    mdf['mvel1_raw'] = mdf['mvel1_raw'].clip(lower=1e-6)
    mdf = mdf.dropna(subset=['pred', 'TARGET', 'mvel1_raw'])
    if len(mdf) < 10:
        return np.nan
    try:
        mdf['decile'] = pd.qcut(mdf['pred'], 10, labels=False, duplicates='drop') + 1
    except Exception:
        return np.nan
    long_  = mdf[mdf['decile'] == mdf['decile'].max()]
    short_ = mdf[mdf['decile'] == mdf['decile'].min()]
    if len(long_) == 0 or len(short_) == 0:
        return np.nan
    vw = lambda g: (g['TARGET'] * g['mvel1_raw']).sum() / g['mvel1_raw'].sum()
    return vw(long_) - vw(short_)


def annualised_sharpe(monthly_series):
    arr = [x for x in monthly_series if not np.isnan(x)]
    if len(arr) < 3 or np.std(arr) == 0:
        return np.nan
    return np.mean(arr) / np.std(arr) * np.sqrt(12)


def load_markets():
    if not NORMALIZED_PATH.exists():
        raise FileNotFoundError(f"Normalized data not found at {NORMALIZED_PATH}")
    return sorted([
        f.name.replace('_ranked.parquet', '').upper()
        for f in NORMALIZED_PATH.glob('*_ranked.parquet')
    ])


def load_data(market):
    market = market.upper()
    path = NORMALIZED_PATH / f'{market}_ranked.parquet'
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_parquet(path)
    df.columns = [
        c.upper() if c.upper() in ('PERMNO', 'DATE', 'TARGET') else c.lower()
        for c in df.columns
    ]
    df['DATE'] = pd.to_datetime(df['DATE'])
    if df['PERMNO'].isna().all():
        df['PERMNO'] = range(len(df))

    if market in MARKET_INFO:
        info        = MARKET_INFO[market]
        start_year  = info['start']
        train_split = info['train']
        valid_split = info['valid']
        end_year    = info['end']
        batch_size  = info['batch_size']
    else:
        years       = df['DATE'].dt.year
        start_year  = int(years.min())
        end_year    = int(years.max())
        total       = end_year - start_year
        train_split = start_year + int(total * 0.6)
        valid_split = start_year + int(total * 0.8)
        batch_size  = min(5000, len(df) // 10)
        print(f"  Using default splits: train≤{train_split}  valid≤{valid_split}  end={end_year}")

    _EXCLUDE = {'PERMNO', 'DATE', 'TARGET', 'mvel1_raw', 'datebk', 'permno', 'date', 'target'}
    feature_cols = [c for c in df.columns if c not in _EXCLUDE]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols, how='any')
    df = df[(df['DATE'].dt.year >= start_year) & (df['DATE'].dt.year <= end_year)]
    df = df.reset_index(drop=True)

    print(f"  {market} | {start_year}–{end_year} | "
          f"train≤{train_split}  valid≤{valid_split} | "
          f"{len(df):,} obs | {len(feature_cols)} features")
    return df, start_year, train_split, valid_split, end_year, batch_size


def in_output(data, include_mvel1_raw=False):
    _EXCLUDE = {'PERMNO', 'DATE', 'TARGET', 'datebk', 'permno', 'date', 'target'}
    if not include_mvel1_raw:
        _EXCLUDE.add('mvel1_raw')
    feature_cols = [c for c in data.columns if c not in _EXCLUDE]
    return data[feature_cols], data['TARGET']


def split_data(final_data, train_split, valid_split, add_year):
    train_end = train_split + add_year
    valid_end = valid_split + add_year
    test_end  = valid_split + add_year + 1

    train = final_data[final_data['DATE'].dt.year <= train_end].copy().reset_index(drop=True)
    valid = final_data[
        (final_data['DATE'].dt.year > train_end) &
        (final_data['DATE'].dt.year <= valid_end)
    ].copy().reset_index(drop=True)
    test = final_data[
        (final_data['DATE'].dt.year > valid_end) &
        (final_data['DATE'].dt.year <= test_end)
    ].copy().reset_index(drop=True)

    print(f"  Split {valid_split + add_year}: "
          f"train {len(train):,}  valid {len(valid):,}  test {len(test):,}", end='  ')
    return train, valid, test


def sharpe_decile_stats(pred_df):
    """Compute monthly EW/VW spread series and derived Sharpe + decile distance."""
    pred_df = pred_df.copy()
    pred_df['DATE'] = pd.to_datetime(pred_df['DATE'])
    pred_df['YearMonth'] = pred_df['DATE'].dt.to_period('M')

    monthly_ew, monthly_vw, decile_dists = [], [], []
    for ym, mdf in pred_df.groupby('YearMonth'):
        if len(mdf) < 10:
            continue
        mdf = mdf.copy()
        try:
            mdf['pred_decile']   = pd.qcut(mdf['pred'],   10, labels=False, duplicates='drop') + 1
            mdf['actual_decile'] = pd.qcut(mdf['TARGET'], 10, labels=False, duplicates='drop') + 1
        except Exception:
            continue
        long_  = mdf[mdf['pred_decile'] == mdf['pred_decile'].max()]
        short_ = mdf[mdf['pred_decile'] == mdf['pred_decile'].min()]
        if len(long_) > 0 and len(short_) > 0:
            monthly_ew.append(long_['TARGET'].mean() - short_['TARGET'].mean())
            monthly_vw.append(_monthly_vw_spread(mdf))
            decile_dists.append(long_['actual_decile'].mean() - short_['actual_decile'].mean())

    sharpe_ew  = annualised_sharpe(monthly_ew)
    sharpe_vw  = annualised_sharpe(monthly_vw)
    decile_dist = np.nanmean(decile_dists) if decile_dists else np.nan
    return sharpe_ew, sharpe_vw, decile_dist


# =============================================================================
# STEP 1 — LOCAL MARKET GBRT MODELS
# =============================================================================

def train_gbrt_year(add_year, market, final_data, train_split, valid_split):
    cur_year   = valid_split + add_year + 1
    start_time = time.time()
    print(f"\n  Year {cur_year}", end='  ', flush=True)

    params_dir = MODEL_DIR / market / 'gbrt'
    os.makedirs(str(params_dir), exist_ok=True)

    train_data, valid_data, test_data = split_data(
        final_data, train_split, valid_split, add_year)
    if len(train_data) == 0 or len(valid_data) == 0 or len(test_data) == 0:
        print("Skipping — insufficient data", end='')
        return None, None

    train_X, train_y = in_output(train_data)
    valid_X, valid_y = in_output(valid_data)
    test_X,  _       = in_output(test_data)

    train_X = np.asarray(train_X);  train_y = np.asarray(train_y)
    valid_X = np.asarray(valid_X);  valid_y = np.asarray(valid_y)
    test_X  = np.asarray(test_X)

    best_mse, best_params, best_model = np.inf, None, None
    for max_depth in PARAM_GRID['max_depth']:
        for lr in PARAM_GRID['learning_rate']:
            for n_est in PARAM_GRID['n_estimators']:
                model = xgb.XGBRegressor(
                    max_depth=max_depth,
                    learning_rate=lr,
                    n_estimators=n_est,
                    objective=XGB_OBJECTIVE,
                    tree_method='hist',
                    random_state=42,
                    verbosity=0,
                )
                model.fit(train_X, train_y)
                mse = mean_squared_error(valid_y, model.predict(valid_X))
                if mse < best_mse:
                    best_mse    = mse
                    best_params = {'max_depth': max_depth, 'learning_rate': lr, 'n_estimators': n_est}
                    best_model  = model

    print(f"depth={best_params['max_depth']}  lr={best_params['learning_rate']}  "
          f"n_est={best_params['n_estimators']}", end='  ')

    dump(best_model, params_dir / f'year{cur_year}.joblib')

    ypred_train = pd.concat([
        train_data[['PERMNO', 'DATE', 'TARGET']],
        pd.DataFrame(best_model.predict(train_X), columns=['pred'])
    ], axis=1)
    ypred_valid = pd.concat([
        valid_data[['PERMNO', 'DATE', 'TARGET']],
        pd.DataFrame(best_model.predict(valid_X), columns=['pred'])
    ], axis=1)
    # include mvel1_raw in test output for VW spread calculation
    test_meta_cols = [c for c in ['PERMNO', 'DATE', 'TARGET', 'mvel1_raw'] if c in test_data.columns]
    ypred_test = pd.concat([
        test_data[test_meta_cols],
        pd.DataFrame(best_model.predict(test_X), columns=['pred'])
    ], axis=1)

    fcast_dir = FORECAST_DIR / market / 'gbrt' / f'Year{cur_year}'
    os.makedirs(str(fcast_dir), exist_ok=True)
    ypred_train.to_csv(fcast_dir / 'trainpred.csv', index=False)
    ypred_valid.to_csv(fcast_dir / 'validpred.csv', index=False)
    ypred_test.to_csv(fcast_dir  / 'testpred.csv',  index=False)

    print(f"{time.time() - start_time:.1f}s", end='', flush=True)
    return ypred_test, best_model


def train_gbrt_years(add_years_list, market, final_data, train_split, valid_split):
    predictions, feat_imps = [], []
    for add_year in add_years_list:
        ypred, model = train_gbrt_year(add_year, market, final_data, train_split, valid_split)
        if ypred is not None and model is not None:
            predictions.append(ypred)
            feat_imps.append(model.feature_importances_)
    if not predictions:
        return pd.DataFrame(), np.array([])
    return pd.concat(predictions), np.array(feat_imps)


def train_gbrt_market(market):
    print('=' * 60)
    print(f'GBRT  {market}')
    print('=' * 60)

    final_data, start_year, train_split, valid_split, end_year, _ = load_data(market)
    if len(final_data) == 0:
        print(f"No data for {market}, skipping.")
        return None

    sample_X, _ = in_output(final_data.head())
    feature_names = list(sample_X.columns)
    print(f"  Grid: depth={PARAM_GRID['max_depth']}  lr={PARAM_GRID['learning_rate']}  "
          f"n_est={PARAM_GRID['n_estimators']}")
    print(f"  Test years: {valid_split + 1} → {end_year}")

    add_years = range(0, end_year - valid_split)
    t0 = time.time()
    pred_df, feat_imps = train_gbrt_years(add_years, market, final_data, train_split, valid_split)
    total_time = time.time() - t0

    if pred_df.empty:
        print(f"No predictions generated for {market}.")
        return None

    ss_res  = ((pred_df['TARGET'] - pred_df['pred']) ** 2).sum()
    ss_tot  = (pred_df['TARGET'] ** 2).sum()
    r2_oos  = 1 - ss_res / ss_tot
    mse     = ss_res / len(pred_df)

    print(f"\n{'=' * 40}")
    print(f"RESULTS  {market}")
    print(f"{'=' * 40}")
    print(f"R² OOS:     {r2_oos:.4f}")
    print(f"MSE:        {mse:.6f}")
    print(f"N test obs: {len(pred_df):,}")
    print(f"Total time: {total_time / 60:.1f} min")

    pred_df['year'] = pd.to_datetime(pred_df['DATE']).dt.year
    for label, mask in [('≤ 2017', pred_df['year'] <= 2017), ('> 2017', pred_df['year'] > 2017)]:
        sub = pred_df[mask]
        if len(sub) == 0:
            print(f"  [{label}]  no observations"); continue
        ss_r = ((sub['TARGET'] - sub['pred']) ** 2).sum()
        ss_t = (sub['TARGET'] ** 2).sum()
        r2_s = 1 - ss_r / ss_t if ss_t != 0 else float('nan')
        print(f"  [{label}]  R² OOS: {r2_s:.4f}   MSE: {ss_r/len(sub):.6f}   N: {len(sub):,}")
    pred_df.drop(columns=['year'], inplace=True)

    save_dir = FORECAST_DIR / market
    os.makedirs(str(save_dir), exist_ok=True)
    pred_df.to_csv(save_dir / 'gbrtpred.csv', index=False)

    top_features = []
    if len(feat_imps) > 0:
        avg_imp  = feat_imps.mean(axis=0)
        feat_df  = pd.DataFrame({'feature': feature_names, 'importance': avg_imp})
        feat_df  = feat_df.sort_values('importance', ascending=False)
        feat_df['importance_pct'] = (feat_df['importance'] / feat_df['importance'].sum() * 100).round(2)
        print("\nTop 10 Features:")
        print(feat_df.head(10).to_string(index=False))
        feat_df.to_csv(save_dir / 'gbrt_feature_importance.csv', index=False)
        top_features = feat_df.head(5)['feature'].tolist()

    return {
        'market':       market,
        'r2_oos':       r2_oos,
        'mse':          mse,
        'n_obs':        len(pred_df),
        'time_minutes': total_time / 60,
        'top_features': top_features,
    }


# =============================================================================
# STEP 2 — US MODEL TRANSFER TO INTERNATIONAL MARKETS
# =============================================================================

def apply_us_model_to_market(market):
    print('=' * 60)
    print(f'APPLYING US GBRT MODEL TO  {market}')
    print('=' * 60)

    if market == 'USA':
        print("Skipping USA — this is the source market.")
        return None

    final_data, start_year, train_split, valid_split, end_year, _ = load_data(market)
    if len(final_data) == 0:
        print(f"No data for {market}, skipping.")
        return None

    us_model_files = sorted(US_MODEL_PATH.glob('year*.joblib'))
    if not us_model_files:
        print("ERROR: No US GBRT models found. Run Step 1 for USA first.")
        return None

    us_model_years = [int(f.stem.replace('year', '')) for f in us_model_files]
    print(f"US models available: {min(us_model_years)}–{max(us_model_years)}")

    intl_test_years = list(range(valid_split + 1, end_year + 1))
    common_years    = [y for y in intl_test_years if y in us_model_years]
    if not common_years:
        print("No overlapping years between US models and market test period.")
        return None

    print(f"Test years: {min(common_years)}–{max(common_years)}")
    t0, all_predictions = time.time(), []

    for year in common_years:
        print(f"  Year {year}", end='  ', flush=True)
        model_path = US_MODEL_PATH / f'year{year}.joblib'
        if not model_path.exists():
            print("model not found, skipping", end='  '); continue
        us_model  = load(model_path)
        year_data = final_data[final_data['DATE'].dt.year == year].copy().reset_index(drop=True)
        if len(year_data) == 0:
            print("no data, skipping", end='  '); continue
        X, _ = in_output(year_data)
        pred = us_model.predict(np.asarray(X))
        pred_df = pd.concat([
            year_data[['PERMNO', 'DATE', 'TARGET']],
            pd.DataFrame(pred, columns=['pred'])
        ], axis=1)
        all_predictions.append(pred_df)
        print(f"{len(year_data):,} obs", end='  ')

    total_time = time.time() - t0
    if not all_predictions:
        print(f"No predictions generated for {market}.")
        return None

    forecast = pd.concat(all_predictions, ignore_index=True)

    ss_res              = ((forecast['TARGET'] - forecast['pred']) ** 2).sum()
    ss_tot              = (forecast['TARGET'] ** 2).sum()
    ss_tot_demeaned     = ((forecast['TARGET'] - forecast['TARGET'].mean()) ** 2).sum()
    r2_oos              = 1 - ss_res / ss_tot
    r2_oos_demeaned     = 1 - ss_res / ss_tot_demeaned
    mse                 = ss_res / len(forecast)
    rank_corr, _        = spearmanr(forecast['TARGET'], forecast['pred'])
    sharpe_ew, sharpe_vw, decile_dist = sharpe_decile_stats(forecast)

    print(f"\n{'=' * 40}")
    print(f"RESULTS  {market}  (US GBRT Model)")
    print(f"{'=' * 40}")
    print(f"R² OOS (paper):    {r2_oos:.4f}")
    print(f"R² OOS (trad):     {r2_oos_demeaned:.4f}")
    print(f"MSE:               {mse:.6f}")
    print(f"Rank Corr:         {rank_corr:.4f}")
    print(f"Sharpe EW:         {sharpe_ew:.2f}" if not np.isnan(sharpe_ew) else "Sharpe EW:  NA")
    print(f"Sharpe VW:         {sharpe_vw:.2f}" if not np.isnan(sharpe_vw) else "Sharpe VW:  NA")
    print(f"Decile Dist:       {decile_dist:.2f}" if not np.isnan(decile_dist) else "Decile Dist: NA")
    print(f"N test obs:        {len(forecast):,}")
    print(f"Test period:       {min(common_years)}–{max(common_years)}")
    print(f"Time:              {total_time:.1f}s")

    forecast['year']      = forecast['DATE'].dt.year
    forecast['YearMonth'] = pd.to_datetime(forecast['DATE']).dt.to_period('M')
    for label, mask in [('≤ 2017', forecast['year'] <= 2017), ('> 2017', forecast['year'] > 2017)]:
        sub = forecast[mask]
        if len(sub) == 0:
            print(f"  [{label}]  no observations"); continue
        ss_r  = ((sub['TARGET'] - sub['pred']) ** 2).sum()
        ss_t  = (sub['TARGET'] ** 2).sum()
        r2_s  = 1 - ss_r / ss_t if ss_t != 0 else float('nan')
        rc_s, _ = spearmanr(sub['TARGET'], sub['pred'])
        sub_ew = []
        for ym, mdf in sub.groupby('YearMonth'):
            if len(mdf) < 10: continue
            mdf = mdf.copy()
            try:
                mdf['pred_decile'] = pd.qcut(mdf['pred'], 10, labels=False, duplicates='drop') + 1
            except Exception: continue
            lg = mdf[mdf['pred_decile'] == mdf['pred_decile'].max()]
            sh = mdf[mdf['pred_decile'] == mdf['pred_decile'].min()]
            if len(lg) > 0 and len(sh) > 0:
                sub_ew.append(lg['TARGET'].mean() - sh['TARGET'].mean())
        sharpe_s = annualised_sharpe(sub_ew)
        print(f"  [{label}]  R²: {r2_s:.4f}   MSE: {ss_r/len(sub):.6f}   "
              f"RankCorr: {rc_s:.4f}   SharpeEW: "
              + (f"{sharpe_s:.2f}" if not np.isnan(sharpe_s) else "NA")
              + f"   N: {len(sub):,}")

    forecast.drop(columns=['year', 'YearMonth'], inplace=True)

    save_dir = FORECAST_DIR / 'USmodel' / market
    os.makedirs(str(save_dir), exist_ok=True)
    forecast.to_csv(save_dir / 'gbrtpred.csv', index=False)

    return {
        'market':           market,
        'r2_oos':           r2_oos,
        'r2_oos_demeaned':  r2_oos_demeaned,
        'mse':              mse,
        'rank_corr':        rank_corr,
        'sharpe_ew':        sharpe_ew,
        'sharpe_vw':        sharpe_vw,
        'decile_dist':      decile_dist,
        'n_obs':            len(forecast),
        'test_start':       min(common_years),
        'test_end':         max(common_years),
        'time_seconds':     total_time,
    }


# =============================================================================
# STEP 3 — METRICS (R² OOS, MSE, Sharpe, Rank Corr, Feature Importance, CKA)
# =============================================================================

def get_feature_importance(market):
    importance_file = FORECAST_DIR / market / 'gbrt_feature_importance.csv'
    if importance_file.exists():
        return pd.read_csv(importance_file)
    model_dir = MODEL_DIR / market / 'gbrt'
    if not model_dir.exists():
        return None
    model_files = list(model_dir.glob('year*.joblib'))
    if not model_files:
        return None
    first_model  = load(model_files[0])
    n_features   = first_model.n_features_in_
    all_imp      = [load(mf).feature_importances_
                    for mf in model_files
                    if load(mf).n_features_in_ == n_features]
    if not all_imp:
        return None
    # derive feature names dynamically from the parquet file
    path = NORMALIZED_PATH / f'{market}_ranked.parquet'
    _EXCL = {'PERMNO', 'DATE', 'TARGET', 'mvel1_raw', 'datebk', 'permno', 'date', 'target'}
    if path.exists():
        cols = pd.read_parquet(path, columns=None).columns.tolist()
        feat_names = [c.lower() for c in cols if c.upper() not in {x.upper() for x in _EXCL}]
    else:
        feat_names = [f'feature_{i}' for i in range(n_features)]
    feat_names = feat_names[:n_features]
    avg_imp = np.mean(all_imp, axis=0)
    imp_df  = pd.DataFrame({'feature': feat_names, 'importance': avg_imp})
    imp_df  = imp_df.sort_values('importance', ascending=False)
    imp_df['importance_pct'] = (imp_df['importance'] / imp_df['importance'].sum() * 100).round(2)
    return imp_df


def calculate_cka_similarity(market, reference_market='USA'):
    if market == reference_market:
        return 1.0
    mkt_dir = MODEL_DIR / market           / 'gbrt'
    ref_dir = MODEL_DIR / reference_market / 'gbrt'
    if not mkt_dir.exists() or not ref_dir.exists():
        return np.nan
    mkt_years = {int(f.stem.replace('year', '')) for f in mkt_dir.glob('year*.joblib')}
    ref_years = {int(f.stem.replace('year', '')) for f in ref_dir.glob('year*.joblib')}
    common    = sorted(mkt_years & ref_years)
    if not common:
        return np.nan
    cka_scores = []
    for year in common:
        mkt_model = load(mkt_dir / f'year{year}.joblib')
        ref_model = load(ref_dir / f'year{year}.joblib')
        mkt_imp = mkt_model.feature_importances_
        ref_imp = ref_model.feature_importances_
        min_len = min(len(mkt_imp), len(ref_imp))
        if min_len == 0:
            continue
        mkt_imp = mkt_imp[:min_len]
        ref_imp = ref_imp[:min_len]
        # Linear CKA on 1-D importance vectors
        hsic_xy = np.dot(mkt_imp, ref_imp)
        hsic_xx = np.dot(mkt_imp, mkt_imp)
        hsic_yy = np.dot(ref_imp, ref_imp)
        if hsic_xx > 0 and hsic_yy > 0:
            cka_scores.append(hsic_xy / np.sqrt(hsic_xx * hsic_yy))
    return float(np.mean(cka_scores)) if cka_scores else np.nan


def compute_metrics_market(market):
    print(f"\n{'=' * 50}")
    print(f"METRICS  {market}")
    print(f"{'=' * 50}")

    pred_file = FORECAST_DIR / market / 'gbrtpred.csv'
    if not pred_file.exists():
        print(f"  No predictions found for {market}. Run Step 1 first.")
        return None

    pred_df = pd.read_csv(pred_file, parse_dates=['DATE'])

    ss_res          = ((pred_df['TARGET'] - pred_df['pred']) ** 2).sum()
    ss_tot          = (pred_df['TARGET'] ** 2).sum()
    ss_tot_demeaned = ((pred_df['TARGET'] - pred_df['TARGET'].mean()) ** 2).sum()
    r2_oos          = 1 - ss_res / ss_tot
    r2_oos_dem      = 1 - ss_res / ss_tot_demeaned
    mse             = ss_res / len(pred_df)
    rank_corr, _    = spearmanr(pred_df['TARGET'], pred_df['pred'])

    sharpe_ew, sharpe_vw, decile_dist = sharpe_decile_stats(pred_df)

    imp_df  = get_feature_importance(market)
    cka_sim = calculate_cka_similarity(market)

    print(f"  R² OOS (paper):   {r2_oos:.4f}")
    print(f"  R² OOS (trad):    {r2_oos_dem:.4f}")
    print(f"  MSE:              {mse:.6f}")
    print(f"  Rank Corr:        {rank_corr:.4f}")
    print(f"  Sharpe EW:        {sharpe_ew:.2f}" if not np.isnan(sharpe_ew) else "  Sharpe EW:  NA")
    print(f"  Sharpe VW:        {sharpe_vw:.2f}" if not np.isnan(sharpe_vw) else "  Sharpe VW:  NA")
    print(f"  Decile Dist:      {decile_dist:.2f}" if not np.isnan(decile_dist) else "  Decile Dist: NA")
    print(f"  CKA vs USA:       {cka_sim:.4f}" if not np.isnan(cka_sim) else "  CKA vs USA:  NA")
    print(f"  N obs:            {len(pred_df):,}")

    if imp_df is not None:
        print("  Top 5 features:")
        print(imp_df.head(5)[['feature', 'importance_pct']].to_string(index=False))

    return {
        'market':       market,
        'R2OOS':        round(r2_oos,      4),
        'R2OOS_trad':   round(r2_oos_dem,  4),
        'MSE':          round(mse,          6),
        'RankCorr':     round(rank_corr,    4),
        'SharpeEW':     round(sharpe_ew,   2) if not np.isnan(sharpe_ew)   else np.nan,
        'SharpeVW':     round(sharpe_vw,   2) if not np.isnan(sharpe_vw)   else np.nan,
        'DecileDist':   round(decile_dist, 2) if not np.isnan(decile_dist) else np.nan,
        'CKA_vs_USA':   round(cka_sim,     4) if not np.isnan(cka_sim)     else np.nan,
        'N':            len(pred_df),
    }


# =============================================================================
# MAIN  —  run all three steps in sequence
# =============================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('ALPHA GO EVERYWHERE — GBRT LOCAL PIPELINE')
    print('Choi, Jiang, Zhang (2025) Replication')
    print('=' * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    markets = load_markets()
    print(f"Markets found: {markets}\n")

    # ------------------------------------------------------------------
    # STEP 1: Train local GBRT per market
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('STEP 1 — LOCAL MARKET MODELS')
    print('=' * 60)
    step1_results = []
    for market in markets:
        try:
            result = train_gbrt_market(market)
            if result is not None:
                step1_results.append(result)
        except Exception as e:
            print(f"ERROR training {market}: {e}")
            import traceback; traceback.print_exc()

    if step1_results:
        s1_df = pd.DataFrame([{
            'Market':   r['market'],
            'R2OOS':    round(r['r2_oos'], 4),
            'MSE':      round(r['mse'],    6),
            'N':        r['n_obs'],
            'Time_min': round(r['time_minutes'], 1),
        } for r in step1_results])
        print('\n' + '=' * 60)
        print('STEP 1 SUMMARY')
        print('=' * 60)
        print(s1_df.to_string(index=False))
        s1_df.to_csv(WORKING_DIR / 'gbrt_results_summary.csv', index=False)
        print(f"\nSaved → {WORKING_DIR / 'gbrt_results_summary.csv'}")

    # ------------------------------------------------------------------
    # STEP 2: Apply US model to international markets
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('STEP 2 — US MODEL TRANSFER')
    print('=' * 60)
    if not US_MODEL_PATH.exists():
        print(f"WARNING: US model folder not found at {US_MODEL_PATH}.")
        print("Skipping Step 2. Ensure Step 1 completed for USA.")
    else:
        intl_markets  = [m for m in markets if m != 'USA']
        step2_results = []
        for market in intl_markets:
            try:
                result = apply_us_model_to_market(market)
                if result is not None:
                    step2_results.append(result)
            except Exception as e:
                print(f"ERROR with {market}: {e}")
                import traceback; traceback.print_exc()

        if step2_results:
            s2_df = pd.DataFrame([{
                'Market':     r['market'],
                'R2OOS':      round(r['r2_oos'],     4),
                'MSE':        round(r['mse'],         6),
                'RankCorr':   round(r['rank_corr'],  4),
                'SharpeEW':   round(r['sharpe_ew'],  2) if not np.isnan(r['sharpe_ew']) else np.nan,
                'SharpeVW':   round(r['sharpe_vw'],  2) if not np.isnan(r['sharpe_vw']) else np.nan,
                'DecileDist': round(r['decile_dist'],2) if not np.isnan(r['decile_dist']) else np.nan,
                'N':          r['n_obs'],
                'TestPeriod': f"{r['test_start']}-{r['test_end']}",
            } for r in step2_results])
            print('\n' + '=' * 60)
            print('STEP 2 SUMMARY — US GBRT ON INTERNATIONAL MARKETS')
            print('=' * 60)
            print(s2_df.to_string(index=False))
            s2_df.to_csv(WORKING_DIR / 'gbrt_USmodel_results_summary.csv', index=False)
            print(f"\nSaved → {WORKING_DIR / 'gbrt_USmodel_results_summary.csv'}")

            # Local vs US model comparison
            local_path = WORKING_DIR / 'gbrt_results_summary.csv'
            if local_path.exists():
                local_df   = pd.read_csv(local_path)
                comparison = []
                for r in step2_results:
                    row = local_df[local_df['Market'] == r['market']]
                    if len(row) > 0:
                        local_r2 = row['R2OOS'].values[0]
                        comparison.append({
                            'Market':      r['market'],
                            'LocalR2':     round(local_r2,    4),
                            'USR2':        round(r['r2_oos'], 4),
                            'R2Diff':      round(local_r2 - r['r2_oos'], 4),
                            'LocalBetter': 'Yes' if local_r2 > r['r2_oos'] else 'No',
                        })
                if comparison:
                    comp_df = pd.DataFrame(comparison)
                    print('\n' + '=' * 60)
                    print('LOCAL vs US MODEL COMPARISON')
                    print('=' * 60)
                    print(comp_df.to_string(index=False))
                    comp_df.to_csv(WORKING_DIR / 'gbrt_local_vs_USmodel_comparison.csv', index=False)

    # ------------------------------------------------------------------
    # STEP 3: Full metrics for all markets
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('STEP 3 — FULL METRICS')
    print('=' * 60)
    step3_results = []
    for market in markets:
        try:
            result = compute_metrics_market(market)
            if result is not None:
                step3_results.append(result)
        except Exception as e:
            print(f"ERROR computing metrics for {market}: {e}")
            import traceback; traceback.print_exc()

    if step3_results:
        metrics_df = pd.DataFrame(step3_results)
        metrics_df.to_csv(WORKING_DIR / 'gbrt_full_metrics.csv', index=False)
        print(f"\nFull metrics saved → {WORKING_DIR / 'gbrt_full_metrics.csv'}")
        print(metrics_df.to_string(index=False))

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")