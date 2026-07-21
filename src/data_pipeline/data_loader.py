import yfinance as yf
import pandas as pd
import os
import sys

# Fix Windows terminal Unicode issue
sys.stdout.reconfigure(encoding='utf-8')

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "UNH",
    "JNJ", "WMT", "XOM", "PG", "MA",
    "HD", "CVX", "MRK", "ABBV", "PEP"
]


def download_stock_data(
    tickers=DEFAULT_TICKERS,
    start="2018-01-01",
    end="2024-12-31",
    interval="1d",
    save_path="data/raw/stock_prices.csv"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"[DataLoader] Downloading {len(tickers)} tickers from {start} to {end}...")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=True
    )

    # ── Check if data is empty ─────────────────────────────────────────────
    if raw.empty:
        raise ValueError(
            "[DataLoader] ERROR: No data downloaded. "
            "Check internet connection or ticker symbols."
        )

    # ── Flatten MultiIndex columns ─────────────────────────────────────────
    # New yfinance returns columns like ('Close', 'AAPL')
    # We flatten to 'Close_AAPL'
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ['_'.join(col).strip() for col in raw.columns.values]
    
    raw.index.name = "Date"

    # ── Remove any tickers that had all NaN (failed downloads) ────────────
    before = raw.shape[1]
    raw.dropna(axis=1, how="all", inplace=True)
    after = raw.shape[1]
    if before != after:
        print(f"[DataLoader] Removed {before - after} empty columns.")

    raw.to_csv(save_path)
    print(f"[DataLoader] Saved to: {save_path} | Shape: {raw.shape}")
    print(f"[DataLoader] Date range: {raw.index[0].date()} to {raw.index[-1].date()}")

    return raw


def load_stock_data(path="data/raw/stock_prices.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[DataLoader] File not found: {path}. "
            f"Run download_stock_data() first."
        )
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    print(f"[DataLoader] Loaded from: {path} | Shape: {df.shape}")
    return df


def extract_close_prices(df, tickers=DEFAULT_TICKERS):
    close_cols = [f"Close_{t}" for t in tickers if f"Close_{t}" in df.columns]

    if not close_cols:
        raise ValueError(
            "[DataLoader] No Close price columns found. "
            "Check column names in stock_prices.csv."
        )

    close_df = df[close_cols].copy()
    close_df.columns = [c.replace("Close_", "") for c in close_df.columns]
    close_df.dropna(how="all", inplace=True)

    print(f"[DataLoader] Extracted close prices for {len(close_df.columns)} tickers.")
    return close_df


if __name__ == "__main__":
    # ── Step 1: Download ───────────────────────────────────────────────────
    raw_df = download_stock_data()

    # ── Step 2: Extract close prices ──────────────────────────────────────
    close_df = extract_close_prices(raw_df)

    # ── Step 3: Print sample ───────────────────────────────────────────────
    print("\nClose Prices (last 3 rows):")
    print(close_df.tail(3).to_string())

    print(f"\nTotal tickers downloaded : {len(close_df.columns)}")
    print(f"Total trading days       : {len(close_df)}")
