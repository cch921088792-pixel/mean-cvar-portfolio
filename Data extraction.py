import datetime as dt
import pandas as pd
import pandas_datareader.data as web

# Universe: 60 Good (≤2010) ETFs (kept as-is; includes duplicates in the original list)
TICKERS = [
    "SPY","IVV","VTI","ITOT","IWM","IJR","IWS","IWN",
    "XLK","XLF","XLE","XLV","XLY","XLI","XLP","XLB",
    "XLU","XLRE","XLC","XLV","XLY","XLI","XLP","XLB",
    "AGG","BND","TLT","SHY","LQD","HYG","IJH","IVW",
    "IVE","MTUM","VLUE","USMV","SPLV","QUAL","IWF","IWD",
    "XBI","XOP","SOXX","XME","XES","SLV","GLD","PPLT",
    "DBA","DBC","USO","UNG","VNQ","IYR","RWR","FNDX",
    "COMT","MDY","MUB","FREL","REM","UPRO"
]

# Start date (inclusive). End date defaults to the most recent available trading day.
START = dt.datetime(2009, 1, 1)

dfs = []
for t in TICKERS:
    stooq_t = t + ".US"
    try:
        s = web.DataReader(stooq_t, "stooq", START)["Close"]
        dfs.append(s.rename(t))
        print(f"✓ {t} 下载成功")
    except Exception as e:
        print(f"✗ {t} 下载失败：{e}")

# Merge all successful series and sort by date ascending
prices = pd.concat(dfs, axis=1).sort_index()

# Save to CSV with index label "Date"
prices.to_csv("prices.csv", index_label="Date")
print(f"\n Saved prices.csv ({prices.shape[0]} row × {prices.shape[1]} List)")