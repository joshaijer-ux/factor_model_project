import numpy as np
import pandas as pd
import os
import warnings
from pathlib import Path
from sklearn.linear_model import LinearRegression, HuberRegressor, Lasso, Ridge
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

# ================================================================
# 1.  Paths
# ================================================================
base            = Path(__file__).parent if '__file__' in dir() else Path.cwd()
normalized_path = base / 'normalized'
forecast_path   = base / 'forecasts'
rawsize_path    = base / 'rawsize'
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
# ================================================================
SPLITS = {
    'USA': (1963, 1979, 1989), 'JPN': (2008, 2010, 2011),
    'CHN': (1999, 2004, 2007), 'IND': (2007, 2010, 2012),
    'KOR': (1997, 2003, 2007), 'HKG': (1997, 2003, 2007),
    'TWN': (2007, 2010, 2012), 'FRA': (1995, 2001, 2005),
    'GBR': (2005, 2008, 2010), 'THA': (1997, 2003, 2007),
    'AUS': (2008, 2010, 2011), 'SGP': (2007, 2010, 2012),
    'SWE': (2001, 2005, 2008), 'ZAF': (1997, 2003, 2007),
    'POL': (2006, 2009, 2011), 'ISR': (2005, 2008, 2010),
    'VNM': (2010, 2012, 2013), 'ITA': (2001, 2005, 2008),
    'TUR': (2006, 2009, 2011), 'CHE': (2002, 2006, 2009),
    'IDN': (2005, 2008, 2010), 'GRC': (2006, 2009, 2011),
    'PHL': (2006, 2009, 2011), 'NOR': (2007, 2010, 2012),
    'LKA': (2010, 2012, 2013), 'DNK': (2007, 2010, 2012),
    'FIN': (2007, 2010, 2012), 'SAU': (2010, 2012, 2013),
    'JOR': (2009, 2011, 2012), 'EGY': (2010, 2012, 2013),
    'ESP': (2011, 2012, 2013), 'KWT': (2012, 2013, 2014),
}

MARKETS = [
    "USA", "JPN", "CHN", "IND", "KOR", "HKG", "TWN", "FRA", "GBR", "THA",
    "AUS", "SGP", "SWE", "ZAF", "POL", "ISR", "VNM", "ITA", "TUR", "CHE",
    "IDN", "GRC", "PHL", "NOR", "LKA", "DNK", "FIN", "SAU", "JOR", "EGY",
    "ESP", "KWT",
]

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

def tune_lasso(train_X, train_y, valid_X, valid_y, alphas):
    
    best_alpha, best_mse = None, np.inf
    prev_coef, prev_intercept = None, None

    for alpha in sorted(alphas, reverse=True):
        m = Lasso(alpha=alpha, max_iter=10_000, tol=1e-4,
                  selection='random', random_state=42)
        if prev_coef is not None:
            m.coef_      = prev_coef.copy()
            m.intercept_ = prev_intercept
        m.fit(train_X, train_y)
        prev_coef, prev_intercept = m.coef_.copy(), m.intercept_
        mse = np.mean((m.predict(valid_X) - valid_y) ** 2)
        if mse < best_mse:
            best_mse, best_alpha = mse, alpha

    m_final = Lasso(alpha=best_alpha, max_iter=10_000, tol=1e-4,
                    selection='random', random_state=42)
    m_final.fit(train_X, train_y)
    return m_final, best_alpha

def export_rawsize(market, df):
    if 'mvel1_raw' in df.columns:
        out    = df[['PERMNO', 'DATE', 'mvel1_raw']].copy()
        out    = out.rename(columns={'mvel1_raw': 'size'})
        source = 'mvel1_raw (raw market equity)'
    else:
        print(f"  [WARN] {market}: mvel1_raw not found — "
              f"falling back to exp(ranked mvel1). Re-run rank_norm to fix.")
        out        = df[['PERMNO', 'DATE', 'mvel1']].copy()
        out['size'] = np.exp(out['mvel1'])
        out        = out[['PERMNO', 'DATE', 'size']]
        source     = 'exp(ranked mvel1) [FALLBACK — inaccurate]'

    out['DATE'] = out['DATE'].dt.strftime('%Y-%m-%d')
    out.to_csv(rawsize_path / f"{market}_rawsize.csv", index=False)
    print(f"  ✓ rawsize saved for {market} using {source}")

# ================================================================
# 6.  Local-trained model
# ================================================================
def run_linear_models(market, label='full', test_start=None, test_end=None):
    print(f"\n{'='*55}\n  {market}  [{label}]\n{'='*55}")

    parquet_file = normalized_path / f"{market}_ranked.parquet"
    if not parquet_file.exists():
        print(f"  [SKIP] Not found: {parquet_file}")
        return

    if market not in SPLITS:
        print(f"  [SKIP] No split defined for {market}")
        return

    df = pd.read_parquet(parquet_file)
    try:
        df['DATE'] = pd.to_datetime(df['DATE'])
    except Exception:
        df['DATE'] = pd.to_datetime(df['DATE'].astype(str), errors='coerce')
    if df['DATE'].isna().all():
        df['DATE'] = pd.to_datetime(df['DATE'].astype(str), errors='coerce')
    df = df.dropna(subset=['DATE'])
    if len(df) == 0:
        print(f"  [SKIP] DATE column is entirely NaT — parquet may be corrupt")
        return
    df = df.sort_values('DATE').reset_index(drop=True)

    rawsize_file = rawsize_path / f"{market}_rawsize.csv"
    if not rawsize_file.exists():
        export_rawsize(market, df)

    feat_all  = [f for f in ALL_FEATURES  if f in df.columns and f != 'mvel1_raw']
    feat_ols3 = [f for f in OLS3_FEATURES if f in df.columns and f != 'mvel1_raw']
    missing   = [f for f in ALL_FEATURES  if f not in df.columns]
    if missing:
        print(f"  [WARN] Missing features: {missing}")

    train_start, train_end, valid_end = SPLITS[market]
    df       = df[df['DATE'].dt.year >= train_start].reset_index(drop=True)
    if len(df) == 0:
        print(f"  [SKIP] No data after train_start={train_start}")
        return
    end_year = int(df['DATE'].dt.year.max())

    t_start = test_start if test_start is not None else valid_end + 1
    t_end   = test_end   if test_end   is not None else end_year
    t_start = max(t_start, valid_end + 1)
    t_end   = min(t_end,   end_year)

    if t_start > t_end:
        print(f"  [SKIP] No test years in window [{t_start}, {t_end}]")
        return

    print(f"  train_start={train_start} | init train ≤{train_end} | "
          f"init valid ≤{valid_end} | test {t_start}–{t_end}")

    preds = {'ols-3': [], 'linear': [], 'lasso': [], 'ridge': []}

    for add_year in range(end_year - valid_end):
        test_year = valid_end + add_year + 1
        if test_year < t_start or test_year > t_end:
            continue

        year_col = df['DATE'].dt.year
        train_df = df[year_col <= train_end + add_year]
        valid_df = df[(year_col >  train_end + add_year) &
                      (year_col <= valid_end + add_year)]
        test_df  = df[(year_col >  valid_end + add_year) &
                      (year_col <= valid_end + add_year + 1)]

        if len(train_df) < 50 or len(valid_df) < 10 or len(test_df) == 0:
            continue

        trX,  try_, _ = get_xy(train_df, feat_all)
        vaX,  vay,  _ = get_xy(valid_df, feat_all)
        teX,  _,    _ = get_xy(test_df,  feat_all)
        trX3, _,    _ = get_xy(train_df, feat_ols3)
        teX3, _,    _ = get_xy(test_df,  feat_ols3)

        base_df = test_df[['PERMNO', 'DATE', 'TARGET']].copy()

        ols3 = LinearRegression().fit(trX3, try_)
        row  = base_df.copy(); row['pred'] = ols3.predict(teX3)
        preds['ols-3'].append(row)

        huber, h_a = tune_and_fit(HuberRegressor, HUBER_ALPHAS, trX, try_,
                                   vaX, vay, epsilon=HUBER_EPSILON, max_iter=200)
        row = base_df.copy(); row['pred'] = huber.predict(teX)
        preds['linear'].append(row)

        lasso, l_a = tune_and_fit(Lasso, LASSO_ALPHAS, trX, try_,
                                   vaX, vay, max_iter=10_000)
        row = base_df.copy(); row['pred'] = lasso.predict(teX)
        preds['lasso'].append(row)

        ridge, r_a = tune_and_fit(Ridge, RIDGE_ALPHAS, trX, try_, vaX, vay)
        row = base_df.copy(); row['pred'] = ridge.predict(teX)
        preds['ridge'].append(row)

        print(f"  {test_year} | n_train={len(train_df):>7,} "
              f"n_test={len(test_df):>5,} | "
              f"Huber α={h_a:.0e}  LASSO α={l_a:.0e}  RIDGE α={r_a:.0e}")

    out_dir = forecast_path / 'local' / label / market
    out_dir.mkdir(parents=True, exist_ok=True)
    for model_name, pred_list in preds.items():
        if not pred_list:
            print(f"  [WARN] No predictions for {model_name}")
            continue
        out_df = pd.concat(pred_list).reset_index(drop=True)
        out_df['TARGET'] *= 100
        out_df['pred']   *= 100
        out_df['DATE']    = pd.to_datetime(out_df['DATE']).dt.strftime('%Y-%m-%d')
        out_df.to_csv(out_dir / f"{model_name}_pred.csv")
        print(f"  ✓ {model_name:7s}: {len(out_df):,} rows → {out_dir / model_name}")


# ================================================================
# 7.  World model
# ================================================================
NON_USA_MARKETS    = [m for m in MARKETS if m != 'USA']
COUNTRY_DUMMY_COLS = [f"dummy_{m}" for m in NON_USA_MARKETS]

WORLD_TRAIN_START = 1963
WORLD_TRAIN_END   = 1979
WORLD_VALID_END   = 1989
WORLD_TEST_START  = 1990


def _load_market_for_world(market):
    """Load, parse dates, return DataFrame or None (no side-effects)."""
    f = normalized_path / f"{market}_ranked.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    try:
        df['DATE'] = pd.to_datetime(df['DATE'])
    except Exception:
        df['DATE'] = pd.to_datetime(df['DATE'].astype(str), errors='coerce')
    df = df.dropna(subset=['DATE']).sort_values('DATE').reset_index(drop=True)
    return df if len(df) > 0 else None


def build_world_panel():
    """
    Pool all 32 markets, subtract global monthly mean from TARGET,
    add 31 country dummies.
    """
    print("\n  Building world panel...")
    frames = []
    for market in MARKETS:
        df = _load_market_for_world(market)
        if df is None:
            continue
        df = df[df['DATE'].dt.year >= WORLD_TRAIN_START].copy()
        df['market'] = market
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel['DATE'] = pd.to_datetime(panel['DATE'])

    # Global excess return (Section 3.2): subtract cross-sectional mean each month
    panel['TARGET'] = (panel['TARGET']
                       - panel.groupby('DATE')['TARGET'].transform('mean'))

    # 31 country dummies via get_dummies (USA = reference, no dummy)
    dummies = pd.get_dummies(panel['market'], prefix='dummy', dtype=np.float32)
    dummies = dummies.drop(columns=['dummy_USA'], errors='ignore')
    for col in COUNTRY_DUMMY_COLS:
        if col not in dummies.columns:
            dummies[col] = np.float32(0)
    panel = pd.concat([panel.reset_index(drop=True),
                       dummies[COUNTRY_DUMMY_COLS]], axis=1)
    panel = panel.sort_values('DATE').reset_index(drop=True)

    feat_all  = [f for f in ALL_FEATURES  + COUNTRY_DUMMY_COLS
                 if f in panel.columns and f != 'mvel1_raw']
    feat_ols3 = [f for f in OLS3_FEATURES + COUNTRY_DUMMY_COLS
                 if f in panel.columns and f != 'mvel1_raw']

    print(f"  World panel: {len(panel):,} rows, "
          f"{panel['DATE'].dt.year.min()}–{panel['DATE'].dt.year.max()}, "
          f"{len(feat_all)} features")

    # Pre-impute once (column-mean) and convert to float32 arrays
    imp = SimpleImputer(strategy='mean')
    X_all  = imp.fit_transform(panel[feat_all].values).astype(np.float32)

    imp3   = SimpleImputer(strategy='mean')
    X_ols3 = imp3.fit_transform(panel[feat_ols3].values).astype(np.float32)

    y_arr      = panel['TARGET'].values.astype(np.float32)
    year_arr   = panel['DATE'].dt.year.values.astype(np.int16)
    mkt_arr    = panel['market'].values
    permno_arr = panel['PERMNO'].values
    date_arr   = panel['DATE'].values

    return (feat_all, feat_ols3, X_all, X_ols3, y_arr,
            year_arr, mkt_arr, permno_arr, date_arr)


def run_world_model():
    print(f"\n{'='*60}\n  WORLD MODEL\n{'='*60}")

    (feat_all, feat_ols3, X_all, X_ols3, y_arr,
     year_arr, mkt_arr, permno_arr, date_arr) = build_world_panel()

    end_year = int(year_arr.max())
    accum    = {market: [] for market in MARKETS}

    for add_year in range(end_year - WORLD_VALID_END):
        test_year = WORLD_VALID_END + add_year + 1
        if test_year < WORLD_TEST_START:
            continue

        tr_mask = year_arr <= (WORLD_TRAIN_END + add_year)
        va_mask = ((year_arr >  (WORLD_TRAIN_END + add_year)) &
                   (year_arr <= (WORLD_VALID_END + add_year)))
        te_mask = year_arr == test_year

        n_tr, n_te = tr_mask.sum(), te_mask.sum()
        if n_tr < 50 or va_mask.sum() < 10 or n_te == 0:
            continue

        trX,  try_  = X_all[tr_mask],  y_arr[tr_mask]
        vaX,  vay   = X_all[va_mask],  y_arr[va_mask]
        teX         = X_all[te_mask]
        trX3        = X_ols3[tr_mask]
        teX3        = X_ols3[te_mask]

        # Fit four models on pooled world data
        ols3 = LinearRegression().fit(trX3, try_)

        huber, h_a = tune_and_fit(HuberRegressor, HUBER_ALPHAS, trX, try_,
                                   vaX, vay, epsilon=HUBER_EPSILON, max_iter=200)

        lasso, l_a = tune_lasso(trX, try_, vaX, vay, LASSO_ALPHAS)

        ridge, r_a = tune_and_fit(Ridge, RIDGE_ALPHAS, trX, try_, vaX, vay)

        # Batch-predict the full test year, split by market after
        p_ols3   = ols3.predict(teX3).astype(np.float32)
        p_linear = huber.predict(teX).astype(np.float32)
        p_lasso  = lasso.predict(teX).astype(np.float32)
        p_ridge  = ridge.predict(teX).astype(np.float32)

        te_mkts    = mkt_arr[te_mask]
        te_permnos = permno_arr[te_mask]
        te_dates   = date_arr[te_mask]
        te_targets = y_arr[te_mask]

        for market in MARKETS:
            m_mask = te_mkts == market
            if not m_mask.any():
                continue
            accum[market].append({
                'PERMNO':      te_permnos[m_mask],
                'DATE':        te_dates[m_mask],
                'TARGET':      te_targets[m_mask],
                'pred_ols3':   p_ols3[m_mask],
                'pred_linear': p_linear[m_mask],
                'pred_lasso':  p_lasso[m_mask],
                'pred_ridge':  p_ridge[m_mask],
            })

        print(f"  {test_year} | n_train={n_tr:>8,} n_test={n_te:>7,} | "
              f"Huber α={h_a:.0e}  LASSO α={l_a:.0e}  RIDGE α={r_a:.0e}")

    MODEL_COLS = {
        'ols-3':  'pred_ols3',
        'linear': 'pred_linear',
        'lasso':  'pred_lasso',
        'ridge':  'pred_ridge',
    }
    print("\n  Saving world model forecasts...")
    for market in MARKETS:
        chunks = accum[market]
        if not chunks:
            print(f"    [WARN] No predictions for {market}")
            continue

        all_permno = np.concatenate([c['PERMNO'] for c in chunks])
        all_date   = np.concatenate([c['DATE']   for c in chunks])
        all_target = np.concatenate([c['TARGET'] for c in chunks])

        out_dir = forecast_path / 'world' / market
        out_dir.mkdir(parents=True, exist_ok=True)

        for model_name, pred_col in MODEL_COLS.items():
            all_pred = np.concatenate([c[pred_col] for c in chunks])
            out_df   = pd.DataFrame({
                'PERMNO': all_permno,
                'DATE':   pd.to_datetime(all_date).strftime('%Y-%m-%d'),
                'TARGET': (all_target * 100).round(6),
                'pred':   (all_pred   * 100).round(6),
            })
            out_df.to_csv(out_dir / f"{model_name}_pred.csv", index=True)

        print(f"    ✓ {market}: {len(all_permno):,} rows")

    print("  ✓ World model done — forecasts/world/<market>/")


# ================================================================
# 8.  Run — local models then world model
# ================================================================
for market in MARKETS:
    try:
        run_linear_models(market, label='full')
        run_linear_models(market, label='pre2018',  test_end=2017)
        run_linear_models(market, label='from2018', test_start=2018)
    except Exception as e:
        print(f"\n  FAILED local {market}: {e}")
        import traceback; traceback.print_exc()

try:
    run_world_model()
except Exception as e:
    print(f"\n  FAILED world model: {e}")
    import traceback; traceback.print_exc()
