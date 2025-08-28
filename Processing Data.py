import numpy as np
import pandas as pd

PRICE_CSV  = "prices.csv"
THRESH     = 0.20
END_DATE   = "2024-12-31"
N_SCEN     = 50_000
SEED       = 42


prices = pd.read_csv(PRICE_CSV, index_col="Date", parse_dates=True)

missing_rate = prices.isna().mean()
keep_cols = missing_rate[missing_rate <= THRESH].index
prices = prices[keep_cols]
prices = prices.loc[:END_DATE]

prices = prices.ffill().bfill()

log_ret = np.log(prices / prices.shift(1)).iloc[1:]

prices.to_csv("prices_clean.csv", index_label="Date")
log_ret.to_csv("log_returns.csv", index_label="Date")

rng = np.random.default_rng(SEED)
idx = rng.choice(log_ret.index, size=N_SCEN, replace=True)
R = log_ret.loc[idx].reset_index(drop=True)
R.index.name = "scenario"
R.to_csv("R.csv")

print("Processing completed：")
print(f"  prices_clean.csv  -> {prices.shape}")
print(f"  log_returns.csv   -> {log_ret.shape}")
print(f"  R.csv             -> {R.shape}")
