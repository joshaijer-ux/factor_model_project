"""
Linear_Models.py
Replication of Choi, Jiang & Zhang (2025) — Linear model baseline
Models: OLS-3 | Huber-OLS ("linear") | LASSO | RIDGE
Splits fixed to paper's Appendix B Table B1 (train_end, valid_end),
test extended to 2024 (vs. paper's 2017).
Reads  : normalized/{market}_ranked.parquet
Writes : forecasts/{market}/{model}_pred.csv
         rawsize/{market}_rawsize.csv
"""

import numpy as np
import pandas as pd
import os
import warnings
from pathlib import Path
from sklearn.linear_model import LinearRegression, HuberRegressor, Lasso, Ridge

warnings.filterwarnings('ignore')

# ================================================================
# 1.  Paths
# ================================================================
base             = Path(__file__).parent if '__file__' in dir() else Path.cwd()
normalized_path  = base / 'normalized'
forecast_path    = base / 'forecasts'
rawsize_path     = base / 'rawsize'
forecast_path.mkdir(parents=True, exist_ok=True)
rawsize_path.mkdir(parents=True, exist_ok=True)

# ================================================================
# 2.  Feature lists
# ================================================================
OLS3_FEATURES = ['mvel1', 'bm', 'mom_12']

ALL_FEATURES = [
    'mom_1',   'mvel1',  'mom_6',   'mom_12',  'chmom_6', 'maxret',
    'indmom_a_12', 'retvol', 'dolvol', 'sp',   'turn',    'bm',
    'ep',      'cfp',    'bm_ia',   'cfp_ia',  'herf',    'mve_ia',
    'lev',     'pctacc', 'stddolvol','stdturn', 'dy',      'salecash',
    'ill',     'acc',    'absacc',  'roe',      'egr',     'agr',
    'cashdebt','lgr',    'sgr',     'chpmia',   'depr',    'cashpr',
]

# ================================================================
# 3.  Hyperparameter grids
# ================================================================
LASSO_ALPHAS  = np.logspace(-3, 3, 7).tolist()
RIDGE_ALPHAS  = np.logspace(-3, 3, 7).tolist()
HUBER_EPSILON = 3.09
HUBER_ALPHAS  = [1e-5, 1e-4, 1e-3]

# ================================================================
# 4.  Hard-coded splits from paper Appendix B Table B1
#     Keys   : market code (matching your parquet filenames)
#     Values : (train_end, valid_end)  — last year of each period
#     Test   : valid_end + 1  →  2024  (extended vs. paper's 2017)
# ================================================================
# (train_start, train_end, valid_end)
SPLITS = {
    'USA': (1963, 1979, 1989),
    'JPN': (2008, 2010, 2011),
    'CHN': (1999, 2004, 2007),
    'IND': (2007, 2010, 2012),
    'KOR': (1997, 2003, 2007),
    'HKG': (1997, 2003, 2007),
    'TWN': (2007, 2010, 2012),
    'FRA': (1995, 2001, 2005),
    'GBR': (2005, 2008, 2010),
    'THA': (1997, 2003, 2007),
    'AUS': (2008, 2010, 2011),
    'SGP': (2007, 2010, 2012),
    'SWE': (2001, 2005, 2008),
    'ZAF': (1997, 2003, 2007),
    'POL': (2006, 2009, 2011),
    'ISR': (2005, 2008, 2010),
    'VNM': (2010, 2012, 2013),
    'ITA': (2001, 2005, 2008),
    'TUR': (2006, 2009, 2011),
    'CHE': (2002, 2006, 2009),
    'IDN': (2005, 2008, 2010),
    'GRC': (2006, 2009, 2011),
    'PHL': (2006, 2009, 2011),
    'NOR': (2007, 2010, 2012),
    'LKA': (2010, 2012, 2013),
    'DNK': (2007, 2010, 2012),
    'FIN': (2007, 2010, 2012),
    'SAU': (2010, 2012, 2013),
    'JOR': (2009, 2011, 2012),
    'EGY': (2010, 2012, 2013),
    'ESP': (2011, 2012, 2013),
    'KWT': (2012, 2013, 2014),
}

# ================================================================
# 5.  Utilities
# ================================================================
def get_xy(df, features):
    available = [f for f in features if f in df.columns]
    return df[available].values, df['TARGET'].values, available

def tune_and_fit(model_cls, alpha_grid, train_X, train_y, valid_X, valid_y,
                 **fixed_kwargs):
    best_alpha, best_mse = None, np.inf
    for alpha in alpha_grid:
        m = model_cls(alpha=alpha, **fixed_kwargs)
        m.fit(train_X, train_y)
        mse = np.mean((m.predict(valid_X) - valid_y) ** 2)
        if mse < best_mse:
            best_mse, best_alpha = mse, alpha
    best_m = model_cls(alpha=best_alpha, **fixed_kwargs)
    best_m.fit(train_X, train_y)
    return best_m, best_alpha

def export_rawsize(market, df):
    out = df[['PERMNO', 'DATE', 'mvel1']].copy()
    out['size'] = np.exp(out['mvel1'])
    out = out[['PERMNO', 'DATE', 'size']]
    out['DATE'] = out['DATE'].dt.strftime('%Y-%m-%d')
    out.to_csv(rawsize_path / f"{market}_rawsize.csv", index=False)

# ================================================================
# 6.  Main function
# ================================================================
def run_linear_models(market):
    print(f"\n{'='*55}\n  {market}\n{'='*55}")

    parquet_file = normalized_path / f"{market}_ranked.parquet"
    if not parquet_file.exists():
        print(f"  [SKIP] Not found: {parquet_file}")
        return

    if market not in SPLITS:
        print(f"  [SKIP] No split defined for {market}")
        return

    df = pd.read_parquet(parquet_file)

    # Robust date parsing — handles datetime, Period, string, int64 nanoseconds
    try:
        df['DATE'] = pd.to_datetime(df['DATE'])
    except Exception:
        df['DATE'] = pd.to_datetime(df['DATE'].astype(str), errors='coerce')

    # If still broken, force via string conversion
    if df['DATE'].isna().all():
        df['DATE'] = pd.to_datetime(df['DATE'].astype(str), errors='coerce')

    df = df.dropna(subset=['DATE'])

    if len(df) == 0:
        print(f"  [SKIP] DATE column is entirely NaT — parquet may be corrupt")
        return

    df = df.sort_values('DATE').reset_index(drop=True)

    if 'mvel1' in df.columns:
        export_rawsize(market, df)

    feat_all = [f for f in ALL_FEATURES if f in df.columns]
    feat_ols3 = [f for f in OLS3_FEATURES if f in df.columns]
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"  [WARN] Missing features: {missing}")

    train_start, train_end, valid_end = SPLITS[market]

    df = df[df['DATE'].dt.year >= train_start].reset_index(drop=True)

    if len(df) == 0:
        print(f"  [SKIP] No data after train_start={train_start}")
        return

    end_year = int(df['DATE'].dt.year.max())

    df = df[df['DATE'].dt.year >= train_start].reset_index(drop=True)

    print(f"  train_start={train_start} | init train ≤{train_end} | "
          f"init valid ≤{valid_end} | test {valid_end + 1}–{end_year}")  # ← use end_year var

    preds = {'ols-3': [], 'linear': [], 'lasso': [], 'ridge': []}

    # rolling window: add_year goes from 0 up to end_year - valid_end - 1
    # each iteration predicts one year of test data (matching authors' split_data)
    for add_year in range(end_year - valid_end):
        train_df = df[df['DATE'].dt.year <= train_end + add_year]
        valid_df = df[
            (df['DATE'].dt.year >  train_end + add_year) &
            (df['DATE'].dt.year <= valid_end + add_year)
        ]
        test_df = df[
            (df['DATE'].dt.year >  valid_end + add_year) &
            (df['DATE'].dt.year <= valid_end + add_year + 1)
        ]
        test_year = valid_end + add_year + 1

        if len(train_df) < 50 or len(valid_df) < 10 or len(test_df) == 0:
            continue  # skip this year only, not the whole market

        trX,  try_, _ = get_xy(train_df, feat_all)
        vaX,  vay,  _ = get_xy(valid_df, feat_all)
        teX,  _,    _ = get_xy(test_df,  feat_all)
        trX3, _,    _ = get_xy(train_df, feat_ols3)
        teX3, _,    _ = get_xy(test_df,  feat_ols3)

        base_df = test_df[['PERMNO', 'DATE', 'TARGET']].copy()

        # OLS-3
        ols3 = LinearRegression().fit(trX3, try_)
        row  = base_df.copy(); row['pred'] = ols3.predict(teX3)
        preds['ols-3'].append(row)

        # Huber-OLS
        huber, h_a = tune_and_fit(
            HuberRegressor, HUBER_ALPHAS, trX, try_, vaX, vay,
            epsilon=HUBER_EPSILON, max_iter=200
        )
        row = base_df.copy(); row['pred'] = huber.predict(teX)
        preds['linear'].append(row)

        # LASSO
        lasso, l_a = tune_and_fit(
            Lasso, LASSO_ALPHAS, trX, try_, vaX, vay, max_iter=10_000
        )
        row = base_df.copy(); row['pred'] = lasso.predict(teX)
        preds['lasso'].append(row)

        # RIDGE
        ridge, r_a = tune_and_fit(
            Ridge, RIDGE_ALPHAS, trX, try_, vaX, vay
        )
        row = base_df.copy(); row['pred'] = ridge.predict(teX)
        preds['ridge'].append(row)

        print(f"  {test_year} | n_train={len(train_df):>7,} "
              f"n_test={len(test_df):>5,} | "
              f"Huber α={h_a:.0e}  LASSO α={l_a:.0e}  RIDGE α={r_a:.0e}")

    # Save forecasts
    out_dir = forecast_path / market
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, pred_list in preds.items():
        if not pred_list:
            print(f"  [WARN] No predictions for {model_name}")
            continue
        out_df = pd.concat(pred_list).reset_index(drop=True)
        out_df['TARGET'] *= 100
        out_df['pred']   *= 100
        out_df['DATE']    = pd.to_datetime(out_df['DATE']).dt.strftime('%Y-%m-%d')
        out_path = out_dir / f"{model_name}_pred.csv"
        out_df.to_csv(out_path)
        print(f"  ✓ {model_name:7s}: {len(out_df):,} rows → {out_path.name}")

# ================================================================
# 7.  Run all 32 markets
# ================================================================
markets = [
    "USA", "JPN", "CHN", "IND", "KOR", "HKG", "TWN", "FRA", "GBR", "THA",
    "AUS", "SGP", "SWE", "ZAF", "POL", "ISR", "VNM", "ITA", "TUR", "CHE",
    "IDN", "GRC", "PHL", "NOR", "LKA", "DNK", "FIN", "SAU", "JOR", "EGY",
    "ESP", "KWT",
]

for market in markets:
    try:
        run_linear_models(market)
    except Exception as e:
        print(f"\n  FAILED {market}: {e}")
        import traceback; traceback.print_exc()