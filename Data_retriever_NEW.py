import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

base    = Path(__file__).parent if '__file__' in dir() else Path.cwd()
raw_dir = base / "DATA" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

# ── COMPUSTAT ────────────────────────────────────────────────────────
data_all = (
    pd.concat([
        pd.read_csv(base / 'funda_global.csv', parse_dates=['datadate']),
        pd.read_csv(base / 'funda_us.csv',     parse_dates=['datadate'])
    ], ignore_index=True)
    .drop_duplicates(subset=['gvkey', 'datadate'])
)
data_all.to_csv(base / 'funda_all.csv', index=False)
print(f"Compustat combined rows: {len(data_all):,}")

# ── JKP GLOBAL FACTOR ────────────────────────────────────────────────
jkp_files = [
    'global_factor_fra.csv',
    'global_factor_swe_ita_che.csv',
    'global_factor_chn.csv',
    'global_factor_usa.csv',
    'global_factor_kor_hkg_tha_zaf.csv',
    'global_factor_remaining.csv',
]

df_all = pd.concat(
    [pd.read_csv(base / f, parse_dates=['eom'], low_memory=False) for f in jkp_files],
    ignore_index=True
)
print(f"Total JKP rows loaded: {len(df_all):,}")

# ── SPLIT AND SAVE — vectorized via groupby ──────────────────────────
paper_countries = set([
    "USA","JPN","CHN","IND","KOR","HKG","TWN","FRA","GBR","THA",
    "AUS","SGP","SWE","ZAF","POL","ISR","VNM","ITA","TUR","CHE",
    "IDN","GRC","PHL","NOR","LKA","DNK","FIN","SAU","JOR","EGY",
    "ESP","KWT"
])

downloaded, failed = [], []
for country, df_country in df_all.groupby('excntry'):
    if country not in paper_countries:
        continue
    try:
        country_dir = raw_dir / country
        country_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(df_country.reset_index(drop=True)),
            country_dir / f"{country}.parquet",
            compression='snappy'
        )
        print(f"  {country}: {len(df_country):,} rows saved")
        downloaded.append(country)
    except Exception as e:
        print(f"  FAILED {country}: {e}")
        failed.append(country)

print(f"\nSaved: {len(downloaded)}/32 | Failed: {failed}")


