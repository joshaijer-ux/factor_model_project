import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

# ================================================================
# 0.  Configuration
# ================================================================
base          = Path(__file__).parent if "__file__" in dir() else Path.cwd()
forecast_path = base / "forecasts"
rawsize_path  = base / "rawsize"
output_path   = base / "performance"
output_path.mkdir(parents=True, exist_ok=True)

MODELS  = ["ols-3", "linear", "lasso", "ridge"]

MARKETS = [
    "USA", "JPN", "CHN", "IND", "KOR", "HKG", "TWN", "FRA", "GBR", "THA",
    "AUS", "SGP", "SWE", "ZAF", "POL", "ISR", "VNM", "ITA", "TUR", "CHE",
    "IDN", "GRC", "PHL", "NOR", "LKA", "DNK", "FIN", "SAU", "JOR", "EGY",
    "ESP", "KWT",
]

N_DECILES = 10

FULL_REGIMES = [
    ("",           "local"),
    ("us_trained", "us_trained"),
    ("world",      "world"),
]

PERIOD_REGIMES = [
    # (subfolder, label, date_filter)
    ("",           "local_pre2018",       lambda df: df["DATE"] <  "2018-01-01"),
    ("",           "local_from2018",      lambda df: df["DATE"] >= "2018-01-01"),
    ("us_trained", "us_trained_pre2018",  lambda df: df["DATE"] <  "2018-01-01"),
    ("us_trained", "us_trained_from2018", lambda df: df["DATE"] >= "2018-01-01"),
    ("world",      "world_pre2018",       lambda df: df["DATE"] <  "2018-01-01"),
    ("world",      "world_from2018",      lambda df: df["DATE"] >= "2018-01-01"),
]

CORE_METRICS = ["R2_OOS", "RankCorr", "SharpeEW", "SharpeVW", "DecileDist"]
FULL_METRICS = ["R2_OOS", "MSE", "RankCorr", "SharpeEW", "SharpeVW", "DecileDist", "N"]

# ================================================================
# 1.  Helpers
# ================================================================
def load_size(market):
    f = rawsize_path / f"{market}_rawsize.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, parse_dates=["DATE"])
    df["DATE"]   = pd.to_datetime(df["DATE"])
    df["PERMNO"] = df["PERMNO"].astype(str)
    return df.rename(columns={"size": "mktcap"})[["PERMNO", "DATE", "mktcap"]]


def sharpe_ratio(monthly_returns):
    s = pd.Series(monthly_returns).dropna()
    if len(s) < 2 or s.std(ddof=1) == 0:
        return np.nan
    return (s.mean() / s.std(ddof=1)) * np.sqrt(12)


def get_folder(subfolder, market):
    if subfolder == "":
        return forecast_path / market
    return forecast_path / subfolder / market


# ================================================================
# 2.  Core metric function
# ================================================================
def compute_metrics(df, size_df=None, date_filter=None,
                    full_mode=False):
    df = df.copy()
    df["DATE"]   = pd.to_datetime(df["DATE"])
    df["PERMNO"] = df["PERMNO"].astype(str)

    if date_filter is not None:
        df = df[date_filter(df)]

    has_vw = size_df is not None
    if has_vw:
        size_df = size_df.copy()
        size_df["PERMNO"] = size_df["PERMNO"].astype(str)
        df = df.merge(size_df, on=["PERMNO", "DATE"], how="left")

    ew_ls, vw_ls, rc_list, dsd_list = [], [], [], []
    all_actual, all_pred = [], []
    n_obs = 0

    for date, grp in df.groupby("DATE"):
        grp = grp.dropna(subset=["TARGET", "pred"])
        if len(grp) < N_DECILES:
            continue

        grp = grp.copy()
        n_obs += len(grp)

        grp["pred_decile"] = pd.qcut(
            grp["pred"], q=N_DECILES, labels=False, duplicates="drop"
        ) + 1
        grp["actual_decile"] = pd.qcut(
            grp["TARGET"], q=N_DECILES, labels=False, duplicates="drop"
        ) + 1

        all_actual.append(grp["TARGET"].values)
        all_pred.append(grp["pred"].values)

        rc, _ = spearmanr(grp["TARGET"], grp["pred"])
        rc_list.append(rc * 100)

        long_mask  = grp["pred_decile"] == N_DECILES
        short_mask = grp["pred_decile"] == 1

        ew_ls.append(
            grp.loc[long_mask,  "TARGET"].mean() -
            grp.loc[short_mask, "TARGET"].mean()
        )

        if has_vw and "mktcap" in grp.columns:
            def vw_ret(mask):
                sub = grp.loc[mask, ["TARGET", "mktcap"]].dropna()
                if len(sub) == 0 or sub["mktcap"].sum() == 0:
                    return np.nan
                w = sub["mktcap"] / sub["mktcap"].sum()
                return (w * sub["TARGET"]).sum()
            vw_ls.append(vw_ret(long_mask) - vw_ret(short_mask))

        dsd_list.append(
            grp.loc[long_mask,  "actual_decile"].mean() -
            grp.loc[short_mask, "actual_decile"].mean()
        )

    sr_ew = sharpe_ratio(ew_ls)  if ew_ls  else np.nan
    sr_vw = sharpe_ratio(vw_ls)  if vw_ls  else np.nan
    rc    = np.nanmean(rc_list)  if rc_list else np.nan
    dsd   = np.nanmean(dsd_list) if dsd_list else np.nan

    if all_actual:
        act    = np.concatenate(all_actual)
        pred   = np.concatenate(all_pred)
        ss_res = np.sum((act - pred) ** 2)
        ss_tot = np.sum(act ** 2)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        mse    = ss_res / len(act)
    else:
        r2 = mse = np.nan

    result = {
        "R2_OOS":     round(r2,    4),
        "RankCorr":   round(rc,    3),
        "SharpeEW":   round(sr_ew, 2),
        "SharpeVW":   round(sr_vw, 2),
        "DecileDist": round(dsd,   2),
    }

    if full_mode:
        result["MSE"]      = round(mse, 6) if not np.isnan(mse) else np.nan
        result["N"]        = n_obs

    return result


# ================================================================
# 3.  Build summary for one regime
# ================================================================
def build_summary(subfolder, label, date_filter=None, full_mode=False):
    print(f"\n{'='*65}\n  Regime: {label}\n{'='*65}")

    rows = []
    metrics_list = FULL_METRICS if full_mode else CORE_METRICS

    for market in MARKETS:
        size_df    = load_size(market)
        mkt_folder = get_folder(subfolder, market)

        for model in MODELS:
            csv_file = mkt_folder / f"{model}_pred.csv"
            if not csv_file.exists():
                print(f"  [SKIP] {market}/{label}/{model}")
                row = {"Market": market, "Model": model, **{m: np.nan for m in metrics_list}}
                if full_mode:
                    row["N"] = 0
                rows.append(row)
                continue

            df = pd.read_csv(csv_file, index_col=0)
            df.columns = df.columns.str.strip()

            metrics = compute_metrics(
                df, size_df=size_df, date_filter=date_filter,
                full_mode=full_mode
            )
            metrics.update({"Market": market, "Model": model})
            rows.append(metrics)

            if full_mode:
                print(
                    f"  {market:4s} | {model:8s} | "
                    f"R2={metrics['R2_OOS']:+.4f}  MSE={metrics['MSE']:.6f}  "
                    f"RC={metrics['RankCorr']:+.3f}%  "
                    f"SR_EW={metrics['SharpeEW']:+.3f}  SR_VW={metrics['SharpeVW']:+.3f}  "
                    f"DSD={metrics['DecileDist']:.2f}  N={metrics['N']}  "
                )
            else:
                print(
                    f"  {market:4s} | {model:8s} | "
                    f"R2={metrics['R2_OOS']:+.4f}  "
                    f"RC={metrics['RankCorr']:+.3f}%  "
                    f"SR_EW={metrics['SharpeEW']:+.3f}  SR_VW={metrics['SharpeVW']:+.3f}  "
                    f"DSD={metrics['DecileDist']:.2f}"
                )

    return pd.DataFrame(rows)


# ================================================================
# 4.  Pivot helpers
# ================================================================
def make_table(df, model, metrics_cols):
    sub = df[df["Model"] == model].set_index("Market")
    sub = sub.reindex(MARKETS).dropna(how="all")
    return sub[metrics_cols]


def print_cross_market_avg(df, label, metrics_cols):
    avg = df.groupby("Model")[metrics_cols].mean().round(4)
    print(f"\n--- Cross-market averages ({label}) ---")
    print(avg.to_string())


# ================================================================
# 5a. Run PERIOD SPLITS first (pre-2018 / from-2018)
# ================================================================
print("\n" + "#"*65)
print("  PERIOD SPLIT RESULTS  (pre-2018 and 2018-onwards)")
print("#"*65)

for subfolder, label, date_filter in PERIOD_REGIMES:
    summary = build_summary(subfolder, label,
                            date_filter=date_filter, full_mode=False)

    for model in MODELS:
        tbl  = make_table(summary, model, CORE_METRICS)
        safe = model.replace("-", "")
        tbl.to_csv(output_path / f"{label}_{safe}.csv")

    summary.to_csv(output_path / f"{label}_all_models.csv", index=False)
    print_cross_market_avg(summary, label, CORE_METRICS)


# ================================================================
# 5b. Run FULL SAMPLE
# ================================================================
print("\n" + "#"*65)
print("  FULL SAMPLE RESULTS")
print("#"*65)

for subfolder, label in FULL_REGIMES:
    summary = build_summary(subfolder, label,
                            date_filter=None, full_mode=True)

    for model in MODELS:
        tbl  = make_table(summary, model, FULL_METRICS)
        safe = model.replace("-", "")
        tbl.to_csv(output_path / f"{label}_{safe}_full.csv")

    summary.to_csv(output_path / f"{label}_all_models_full.csv", index=False)
    print_cross_market_avg(summary, label, CORE_METRICS)


print("\n✓ Done — results saved to performance/")