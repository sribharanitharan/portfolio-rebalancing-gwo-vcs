import pandas as pd
import numpy as np
import os
import sys

# Fix module path and Windows terminal encoding
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')

from data_pipeline.data_loader import extract_close_prices, load_stock_data, DEFAULT_TICKERS


def compute_returns(
    close_df,
    method="log",
    save_path="data/processed/returns.csv"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if method == "log":
        returns = np.log(close_df / close_df.shift(1))
    else:
        returns = close_df.pct_change()

    returns.dropna(how="all", inplace=True)
    returns.to_csv(save_path)
    print(f"[Preprocessor] Returns ({method}) saved to: {save_path} | Shape: {returns.shape}")
    return returns


def compute_covariance_matrix(
    returns,
    annualize=True,
    trading_days=252,
    save_path="data/processed/covariance_matrix.csv"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cov = returns.cov()
    if annualize:
        cov = cov * trading_days

    cov.to_csv(save_path)
    print(f"[Preprocessor] Covariance matrix saved to: {save_path} | Shape: {cov.shape}")
    return cov


def compute_expected_returns(
    returns,
    method="mean",
    trading_days=252
):
    if method == "ewm":
        expected = returns.ewm(span=60).mean().iloc[-1] * trading_days
    else:
        expected = returns.mean() * trading_days
    return expected


# ── Feature Engineering ────────────────────────────────────────────────────────

def _rsi(close, period=14):
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_features_for_ticker(ohlcv_df, ticker):
    try:
        close  = ohlcv_df[f"Close_{ticker}"]
        high   = ohlcv_df[f"High_{ticker}"]
        low    = ohlcv_df[f"Low_{ticker}"]
        volume = ohlcv_df[f"Volume_{ticker}"]
    except KeyError:
        print(f"[FeatureEng] Missing columns for {ticker}, skipping.")
        return pd.DataFrame()

    feat = pd.DataFrame(index=ohlcv_df.index)

    # Returns
    for lag in [1, 5, 10, 21]:
        feat[f"{ticker}_ret_{lag}d"] = close.pct_change(lag)

    # Volatility
    daily_ret = close.pct_change()
    feat[f"{ticker}_vol_21d"] = (
        daily_ret.rolling(21, min_periods=15).std() * np.sqrt(252)
    )

    # RSI
    feat[f"{ticker}_rsi_14"] = _rsi(close, period=14)

    # MACD
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feat[f"{ticker}_macd"]        = macd
    feat[f"{ticker}_macd_signal"] = signal
    feat[f"{ticker}_macd_hist"]   = macd - signal

    # Bollinger Bands
    sma20 = close.rolling(20, min_periods=15).mean()
    std20 = close.rolling(20, min_periods=15).std()
    feat[f"{ticker}_bb_width"] = (2 * std20) / sma20
    feat[f"{ticker}_bb_pctb"]  = (close - (sma20 - 2*std20)) / (4*std20 + 1e-9)

    # ATR %
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    feat[f"{ticker}_atr_pct"] = atr14 / (close + 1e-9)

    # Volume Z-score
    vol_mean = volume.rolling(60, min_periods=40).mean()
    vol_std  = volume.rolling(60, min_periods=40).std()
    feat[f"{ticker}_vol_zscore"] = (volume - vol_mean) / (vol_std + 1e-9)

    # Price position in range
    for window in [10, 50]:
        roll_high = high.rolling(window).max()
        roll_low  = low.rolling(window).min()
        denom     = roll_high - roll_low
        feat[f"{ticker}_price_pos_{window}d"] = np.where(
            denom < 1e-8, 0.5, (close - roll_low) / denom
        )

    # Momentum
    feat[f"{ticker}_momentum_21d"] = close / close.shift(21) - 1
    feat[f"{ticker}_momentum_63d"] = close / close.shift(63) - 1

    # Shift to prevent lookahead bias
    feat = feat.shift(1)
    return feat


def build_all_features(
    raw_df,
    tickers=DEFAULT_TICKERS,
    save_path="data/processed/features.csv"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    all_features = []
    for ticker in tickers:
        ticker_feat = build_features_for_ticker(raw_df, ticker)
        if not ticker_feat.empty:
            all_features.append(ticker_feat)

    if not all_features:
        raise ValueError("[FeatureEng] No features generated. Check column names.")

    features_df = pd.concat(all_features, axis=1)
    features_df.dropna(how="all", inplace=True)
    features_df.to_csv(save_path)
    print(f"[Preprocessor] Features saved to: {save_path} | Shape: {features_df.shape}")
    return features_df


if __name__ == "__main__":
    # ── Must run from project root ─────────────────────────────────────────
    # cd "C:\clg\8th sem\8th sem projects\MHO\portfolio-rebalancing-gwo-vcs"

    raw_df   = load_stock_data("data/raw/stock_prices.csv")
    close_df = extract_close_prices(raw_df)

    print("\n[Step 1] Computing returns...")
    returns  = compute_returns(close_df, method="log")

    print("\n[Step 2] Computing covariance matrix...")
    cov_mat  = compute_covariance_matrix(returns)

    print("\n[Step 3] Computing expected returns...")
    exp_ret  = compute_expected_returns(returns)

    print("\n[Step 4] Building ML features...")
    features = build_all_features(raw_df)

    print("\n--- Expected Annual Returns ---")
    print(exp_ret.round(4).to_string())

    print("\n--- Feature Sample (last 2 rows, first 5 cols) ---")
    print(features.iloc[-2:, :5].to_string())

    print("\n[Preprocessor] All steps complete.")
    print(f"  Returns shape  : {returns.shape}")
    print(f"  Cov mat shape  : {cov_mat.shape}")
    print(f"  Features shape : {features.shape}")
