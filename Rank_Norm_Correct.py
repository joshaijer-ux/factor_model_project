import pandas as pd
import numpy as np
import os
from os.path import join
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================
# 1. Path Setup
# Define paths to input (processed) and output (normalized) data
# =============================================================
from pathlib import Path

base           = Path(__file__).parent if '__file__' in dir() else Path.cwd()
data_path      = str(base / "DATA" / "raw")
output_path    = str(base / "normalized")
compustat_path = str(base / "funda_all.csv")
os.makedirs(output_path, exist_ok=True)

print("Input path:", data_path)
print("Output path:", output_path)

# =============================================================
# 2. Loading Compustat & VW market return
# =============================================================
compustat = pd.read_csv(compustat_path, parse_dates=['datadate'], low_memory=False)
compustat['gvkey'] = compustat['gvkey'].astype(str).str.zfill(6)
compustat = compustat.drop_duplicates(subset=['gvkey', 'datadate'])
compustat = compustat.sort_values(['gvkey', 'datadate'])
print(f"Compustat loaded: {len(compustat):,} rows")

mkt_vw = pd.read_csv(base / 'mkt_vw.csv', parse_dates=['date'], dayfirst=True)
mkt_vw = mkt_vw[['location', 'date', 'ret']].copy()
mkt_vw['location'] = mkt_vw['location'].str.upper()   # usa → USA
mkt_vw['date'] = pd.to_datetime(mkt_vw['date']).dt.to_period('M').dt.to_timestamp()
mkt_vw = mkt_vw.rename(columns={'location': 'excntry', 'date': 'DATE', 'ret': 'mkt_vw'})
print(f"Market returns loaded: {len(mkt_vw):,} rows | Countries: {mkt_vw['excntry'].nunique()}")

# =============================================================
# 3. Column Mapping
# Map WRDS/JKP column names to the paper's variable names
# =============================================================
JKP_columns_dic = {
    'permno':            'PERMNO',
    'eom':               'DATE',
    'gvkey':             'gvkey',
    'ret_1_0':           'mom_1',
    'market_equity':     'mvel1',
    'ret_6_1':           'mom_6',
    'ret_12_1':          'mom_12',
    'ret_12_7':          'chmom_6',
    'rmax1_21d':         'maxret',
    'rvol_21d':          'retvol',
    'dolvol_126d':       'dolvol',
    'sale_me':           'sp',
    'turnover_126d':     'turn',
    'be_me':             'bm',
    'ni_me':             'ep',
    'fcf_me':            'cfp',
    'at_be':             'lev',
    'taccruals_ni':      'pctacc',
    'dolvol_var_126d':   'stddolvol',
    'turnover_var_126d': 'stdturn',
    'div12m_me':         'dy',
    'ami_126d':          'ill',
    'ni_be':             'roe',
    'be_gr1a':           'egr',
    'at_gr1':            'agr',
    'ocf_debt':          'cashdebt',
    'debtlt_gr1a':       'lgr',
    'sale_gr1':          'sgr',
    'ebit_sale':         'chpmia',
    'oaccruals_at':      'acc',
    'ff49':              'ff49',
    'assets':            'assets',
    'cash_at':           'cash_at',
    'sales':             'sales',
}

variables = [
    'PERMNO', 'DATE', 'TARGET',
    'mom_1', 'mvel1', 'mom_6', 'mom_12', 'chmom_6', 'maxret',
    'indmom_a_12', 'retvol', 'dolvol', 'sp', 'turn', 'bm', 'ep',
    'cfp', 'bm_ia', 'cfp_ia', 'herf', 'mve_ia', 'lev', 'pctacc',
    'stddolvol', 'stdturn', 'dy', 'salecash', 'ill', 'acc', 'absacc',
    'roe', 'egr', 'agr', 'cashdebt', 'lgr', 'sgr', 'chpmia',
    'depr', 'cashpr'   # FIX: added from Compustat
]

print("Number of model variables (excl. PERMNO, DATE, TARGET):", len(variables) - 3)

# =============================================================
# 4. Leave-one-out
# Implementing LOO to avoid look-ahead self-contamination
# =============================================================
def leave_one_out_mean(df, var, by_cols):
    x         = pd.to_numeric(df[var], errors='coerce')
    grp       = x.groupby([df[c] for c in by_cols])
    grp_sum   = grp.transform('sum')
    grp_count = grp.transform('count')
    loo = pd.Series(np.nan, index=df.index)
    mask = x.notna() & (grp_count > 1)
    loo.loc[mask] = (grp_sum[mask] - x[mask]) / (grp_count[mask] - 1)
    return loo

# =============================================================
# 4b. Winsorization
# Paper: "We winsorize raw returns at the top and bottom 2.5%
#         in each exchange in each month"
# =============================================================
def winsorize_returns(df, col='mom_1', lower=0.025, upper=0.975):
    """
    Winsorize a return column at the top and bottom percentiles
    within each cross-section (DATE).
    """
    df = df.copy()
    def _clip_group(x):
        lo = x.quantile(lower)
        hi = x.quantile(upper)
        return x.clip(lower=lo, upper=hi)
    df[col] = df.groupby('DATE')[col].transform(_clip_group)
    return df

###. def winsorize_returns(df, col='mom_1', lower=0.025, upper=0.975):
    """
    Winsorize a return column at the top and bottom percentiles
    within each cross-section (DATE).
    """
    df = df.copy()
    def _clip_group(x):
        lo = x.quantile(lower)
        hi = x.quantile(upper)
        return x.clip(lower=lo, upper=hi)
    df[col] = df.groupby('DATE')[col].transform(_clip_group)
    return df


# =============================================================
# 5. Target Variable Construction
# TARGET = next month return excess of cross-sectional mean
# Uses permno as stock identifier, falls back to gvkey if permno is null
# =============================================================
def construct_target(df, market):
    id_col = 'gvkey' if df['PERMNO'].isnull().all() else 'PERMNO'
    df = df.copy().sort_values([id_col, 'DATE'])
    df['TARGET'] = df.groupby(id_col)['mom_1'].shift(-1)

    vw = mkt_vw[mkt_vw['excntry'] == market][['DATE', 'mkt_vw']]
    if len(vw) == 0:
        print(f"  WARNING: No VW return for {market}, using EW fallback")
        df['TARGET'] = df['TARGET'] - df.groupby('DATE')['TARGET'].transform('mean')
    else:
        df = df.merge(vw, on='DATE', how='left')
        missing = df['mkt_vw'].isnull().sum()
        if missing > 0:
            print(f"  WARNING: {missing} rows missing VW return, filling with EW mean")
            ew = df.groupby('DATE')['TARGET'].transform('mean')
            df['mkt_vw'] = df['mkt_vw'].fillna(ew)
        df['TARGET'] = df['TARGET'] - df['mkt_vw']
        df = df.drop(columns=['mkt_vw'])
    return df

# =============================================================
# 6. Variable Construction
# Construct paper-style variables
# =============================================================
def construct_variables(df):
    df = df.copy()
    id_col = 'gvkey' if df['PERMNO'].isnull().all() else 'PERMNO'
    before = len(df)
    df = df.sort_values([id_col, 'DATE']).drop_duplicates(subset=[id_col, 'DATE'])
    if len(df) < before:
        print(f"  [construct_variables] Dropped {before - len(df):,} duplicate (ID, DATE) rows")

    # Log market cap
    df['mvel1'] = np.log(df['mvel1'].clip(lower=1e-6))

    # chmom_6 = ret_6_1 - ret_12_7
    df['chmom_6'] = df['mom_6'] - df['chmom_6']

    # absacc = |acc|
    df['absacc'] = df['acc'].abs()

    # salecash = sales / (cash_at * assets)
    df['salecash'] = df['sales'] / (df['cash_at'] * df['assets'])
    df['salecash'] = df['salecash'].replace([np.inf, -np.inf], np.nan)

    # FIX: Leave-one-out industry-adjusted variables (no self-contamination)
    group_cols = ['DATE', 'ff49']
    df['bm_ia']  = df['bm']    - leave_one_out_mean(df, 'bm',    group_cols)
    df['cfp_ia'] = df['cfp']   - leave_one_out_mean(df, 'cfp',   group_cols)
    df['mve_ia'] = df['mvel1'] - leave_one_out_mean(df, 'mvel1', group_cols)

    # FIX: Leave-one-out industry momentum
    df['indmom_a_12'] = leave_one_out_mean(df, 'mom_12', group_cols)

    # Herfindahl index of sales within FF49 x month
    def herfindahl(x):
        total = x.sum()
        if total <= 0:
            return pd.Series(np.nan, index=x.index)
        shares = x / total
        h = (shares ** 2).sum()
        return pd.Series(h, index=x.index)

    df['herf'] = df.groupby(group_cols)['sales'].transform(
        lambda x: herfindahl(x))


    df = df.sort_values([id_col, 'DATE'])
    lag12 = df[[id_col, 'DATE', 'chpmia']].copy()

    lag12['DATE'] = (lag12['DATE'].values.astype('datetime64[M]')
                     + np.timedelta64(12, 'M')).astype('datetime64[ns]')
    lag12['DATE'] = pd.to_datetime(lag12['DATE'])
    lag12 = lag12.drop_duplicates(subset=[id_col, 'DATE'])
    lag12 = lag12.rename(columns={'chpmia': 'chpmia_lag12'})

    df = df.merge(lag12, on=[id_col, 'DATE'], how='left')
    df['chpmia'] = df['chpmia'] - df['chpmia_lag12']
    df = df.drop(columns=['chpmia_lag12'])
    df['chpmia'] = df['chpmia'] - leave_one_out_mean(df, 'chpmia', group_cols)

    return df

# =============================================================
# 7. Missing Value Handling and Rank Normalization
# Fill NAs with monthly median, then rank normalize to [-1, 1]
# =============================================================

def Handle_NA(df):
    feature_cols = [c for c in df.columns if c not in ['PERMNO', 'DATE', 'TARGET']]
    for col in feature_cols:  # column-by-column — avoids wide matrix allocation
        df[col] = df.groupby('DATE')[col].transform(lambda x: x.fillna(x.median()))
    return df


def rank_normalize_col(x):
    """Same logic as rank_column but NaN-safe for transform (preserves index length)"""
    valid = x.notna()
    n = valid.sum()
    if n == 0:
        return x
    result = x.copy()
    ranked = x[valid].rank()
    ranked -= ranked.mean()
    if n > 1:
        ranked /= ((n - 1) / 2)
    result[valid] = ranked
    return result


def rank_all(df):
    need_process = [c for c in df.columns if c not in ['PERMNO', 'TARGET', 'DATE']]
    for col in need_process:                                      # one column at a time
        df[col] = df.groupby('DATE')[col].transform(rank_normalize_col)
    return df


def Rows_Greater_Level(df, level=100):
    counts = df.groupby('DATE').size()
    valid = counts[counts >= level]
    if len(valid) == 0:
        return None, None, 0
    return valid.index.min(), valid.index.max(), len(valid)

print("NA handling and rank normalization functions defined successfully")

# =============================================================
# 8. Main Processing Function
# Loads each market's parquet file, applies all transformations,
# rank-normalizes, and saves the output
# =============================================================
def Rankise(market):
    print(f"\nProcessing {market}...")

    file_path = join(data_path, f"{market}.parquet")
    if not os.path.exists(file_path):
        file_path = join(data_path, market, f"{market}.parquet")
    if not os.path.exists(file_path):
        print(f"  ERROR: File not found: {file_path}")
        return

    raw_data = pd.read_parquet(file_path)
    print(f"  Loaded: {raw_data.shape[0]:,} rows, {raw_data.shape[1]} columns")

    raw_data = raw_data.rename(columns=JKP_columns_dic)
    raw_data['DATE'] = pd.to_datetime(raw_data['DATE']).dt.to_period('M').dt.to_timestamp()
    if 'PERMNO' not in raw_data.columns or raw_data['PERMNO'].isna().all():
        gv = pd.to_numeric(raw_data['gvkey'], errors='coerce')
        raw_data['PERMNO'] = gv.map(
            lambda x: str(int(x)).zfill(6) if pd.notna(x) else np.nan
        )

    # ── Winsorize raw returns at 2.5% / 97.5% per month ────────────────
    raw_data = winsorize_returns(raw_data, col='mom_1', lower=0.025, upper=0.975)
    print("  Winsorized mom_1 at 2.5% / 97.5% per month")

    # ── FIX: Return filters ──────────────────────────────────────────────
    id_col = 'gvkey' if raw_data['PERMNO'].isnull().all() else 'PERMNO'
    raw_data = raw_data.sort_values([id_col, 'DATE'])
    extreme = raw_data['mom_1'].abs() > 3.0
    prev_ret = raw_data.groupby(id_col)['mom_1'].shift(1)
    prev_extreme = prev_ret.abs() > 3.0
    reversal = prev_extreme & (raw_data['mom_1'] * prev_ret < 0)

    # Apply filters
    raw_data.loc[extreme, 'mom_1'] = np.nan  # >300% filter
    raw_data.loc[reversal, 'mom_1'] = np.nan  # next-month reversal after extreme
    raw_data.loc[raw_data['mom_1'] == 0, 'mom_1'] = np.nan  # zero-return filter

    # ── FIX: Merge Compustat for depr and cashpr ─────────────────────────
    raw_data['gvkey'] = (
        pd.to_numeric(raw_data['gvkey'], errors='coerce')  # '4449.0' → 4449.0
        .apply(lambda x: str(int(x)).zfill(6) if pd.notna(x) else np.nan)  # → '004449'
    )

    raw_data = pd.merge_asof(
        raw_data.sort_values('DATE'),
        compustat[['gvkey', 'datadate', 'dp', 'ppent', 'dltt']].sort_values('datadate'),
        left_on='DATE',
        right_on='datadate',
        by='gvkey',
        direction='backward',
        tolerance=pd.Timedelta(days=548) # 18 months
    )


    raw_data['depr']   = (raw_data['dp'] / raw_data['ppent']).replace([np.inf, -np.inf], np.nan)
    raw_data['cashpr'] = (
        (raw_data['mvel1'] + raw_data['dltt'] - raw_data['assets'])
        / (raw_data['cash_at'] * raw_data['assets'])
    ).replace([np.inf, -np.inf], np.nan)

    # ── Construct TARGET and variables ───────────────────────────────────
    raw_data = construct_target(raw_data, market)
    print("  TARGET constructed")

    raw_data = construct_variables(raw_data)
    print("  Paper variables constructed")

    raw_data = raw_data.replace([np.inf, -np.inf], np.nan)

    available_vars = [v for v in variables if v in raw_data.columns]
    missing_vars   = [v for v in variables if v not in raw_data.columns]
    if missing_vars:
        print(f"  Warning: missing variables: {missing_vars}")

    df = raw_data[available_vars].copy()
    df = df.dropna(subset=['TARGET'])
    print(f"  After dropping missing TARGET: {df.shape[0]:,} rows")

    df = Handle_NA(df)

    critical_cols = ['TARGET', 'mom_1', 'mvel1', 'DATE']
    df = df.dropna(subset=critical_cols)
    print(f"  After dropping missing critical columns: {df.shape[0]:,} rows")

    start_month, end_month, valid_months = Rows_Greater_Level(df)
    if valid_months < 36:
        print(f"  WARNING: {market} has only {valid_months} valid months — skipping!")
        return
    print(f"  Valid months (>=100 stocks): {valid_months}")
    print(f"  Date range: {start_month} to {end_month}")

    df = df[(df['DATE'] >= '1963-01-01') & (df['DATE'] <= '2025-12-01')]
    df = df.dropna(subset=critical_cols).reset_index(drop=True)

    print("  Applying rank normalization...")
    rank_all(df)
    df = df.dropna(how='any')
    df = df.reset_index(drop=True)

    print(f"  Final shape: {df.shape}")

    out_path = join(output_path, f"{market}_ranked.parquet")
    df.to_parquet(out_path, index=False)
    print(f"  Saved to: {out_path}")
    print(f"  {market} done!")

# =============================================================
# 9. Run All Markets
# Runs normalization for all 32 markets
# Note: this will take significant time
# =============================================================
markets = [
    "USA", "JPN", "CHN", "IND", "KOR", "HKG", "TWN", "FRA", "GBR", "THA",
    "AUS", "SGP", "SWE", "ZAF", "POL", "ISR", "VNM", "ITA", "TUR", "CHE",
    "IDN", "GRC", "PHL", "NOR", "LKA", "DNK", "FIN", "SAU", "JOR", "EGY",
    "ESP", "KWT"
]

for market in markets:
    try:
        Rankise(market)
    except Exception as e:
        print(f"  FAILED {market}: {e}")

# =============================================================
# 10. Verification - Check all markets processed correctly
# =============================================================

import os

normalized_files = [f for f in os.listdir(output_path) if f.endswith('_ranked.parquet')]
print(f"Number of markets processed: {len(normalized_files)}")
print("\nProcessed markets:")
for f in sorted(normalized_files):
    df_temp = pd.read_parquet(os.path.join(output_path, f))
    market_name = f.replace('_ranked.parquet', '')
    print(f"  {market_name}: {df_temp.shape[0]:,} rows | "
          f"Date range: {df_temp['DATE'].min().date()} to {df_temp['DATE'].max().date()} | "
          f"Missing values: {df_temp.isnull().sum().sum()}")
