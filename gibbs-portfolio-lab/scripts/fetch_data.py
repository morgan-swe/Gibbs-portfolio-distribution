"""Download daily adjusted closes for the 11 sector ETFs into data/prices.csv."""
from datetime import date
from pathlib import Path

import yfinance as yf

TICKERS = ["XLK", "XLV", "XLE", "XLF", "XLU", "XLP", "XLY", "XLI", "XLB", "XLRE", "XLC"]
OUT = Path(__file__).resolve().parent.parent / "data" / "prices.csv"

df = yf.download(TICKERS, start="2018-01-01", end=date.today().isoformat(), auto_adjust=True, progress=False)["Close"]
missing = [t for t in TICKERS if t not in df or df[t].dropna().empty]
if missing:
    raise SystemExit(f"Download failed for: {missing}")
df = df[TICKERS].dropna(how="all")
df.index.name = "date"
OUT.parent.mkdir(exist_ok=True)
df.round(4).to_csv(OUT)
print(f"Wrote {len(df)} rows to {OUT} ({df.index[0].date()} to {df.index[-1].date()})")
