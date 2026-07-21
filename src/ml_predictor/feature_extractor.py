import numpy as np
import pandas as pd


class FeatureExtractor:
    """
    Feature Extractor
    ──────────────────
    Converts raw features.csv into ML-ready input.

    Job:
        - Takes the features.csv (technical indicators per stock)
        - Aggregates per-stock features into a single row per stock
        - Returns a feature matrix: shape (n_assets, n_features)

    This matrix is fed into RegionPredictor to score each asset.
    """

    def __init__(
        self,
        features_path: str = "data/processed/features.csv",
        returns_path: str  = "data/processed/returns.csv"
    ):
        self.features_path = features_path
        self.returns_path  = returns_path
        self.feature_matrix = None
        self.asset_names    = None

    def load(self):
        """Loads features.csv and returns.csv from disk."""
        self.features_df = pd.read_csv(
            self.features_path, index_col="Date", parse_dates=True
        )
        self.returns_df = pd.read_csv(
            self.returns_path, index_col="Date", parse_dates=True
        )
        print(f"[FeatureExtractor] Loaded features: {self.features_df.shape}")
        print(f"[FeatureExtractor] Loaded returns : {self.returns_df.shape}")
        return self

    def _get_ticker_columns(self, ticker: str) -> list:
        """Returns all feature column names belonging to a ticker."""
        return [c for c in self.features_df.columns if c.startswith(f"{ticker}_")]

    def build_asset_feature_matrix(
        self,
        lookback_days: int = 60
    ) -> pd.DataFrame:
        """
        Builds one feature vector per asset using the last N days.

        Aggregation per feature column:
            - mean   → average trend
            - std    → volatility of indicator
            - last   → most recent value

        Returns:
            feature_matrix : DataFrame — shape (n_assets, n_agg_features)
        """
        # Use last `lookback_days` rows
        recent = self.features_df.iloc[-lookback_days:]

        # Get unique tickers from column names
        tickers = list(self.returns_df.columns)
        self.asset_names = tickers

        rows = []
        for ticker in tickers:
            cols = self._get_ticker_columns(ticker)
            if not cols:
                # Fill zeros if ticker has no features
                rows.append(np.zeros(len(cols) * 3 if cols else 10))
                continue

            ticker_data = recent[cols].dropna()
            if ticker_data.empty:
                rows.append(np.zeros(len(cols) * 3))
                continue

            mean_vals = ticker_data.mean().values
            std_vals  = ticker_data.std().fillna(0).values
            last_vals = ticker_data.iloc[-1].values

            # Concatenate mean + std + last for each feature
            row = np.concatenate([mean_vals, std_vals, last_vals])
            rows.append(row)

        # Pad rows to same length (in case some tickers have fewer features)
        max_len = max(len(r) for r in rows)
        rows    = [
            np.pad(r, (0, max_len - len(r)), constant_values=0) for r in rows
        ]

        self.feature_matrix = pd.DataFrame(
            rows,
            index=tickers,
            columns=[f"feat_{i}" for i in range(max_len)]
        )

        print(f"[FeatureExtractor] Asset feature matrix: {self.feature_matrix.shape}")
        return self.feature_matrix

    def build_training_data(
        self,
        forward_days: int = 21,
        lookback_days: int = 60
    ):
        """
        Builds labeled training data for RegionPredictor.

        Label = 1 if asset's forward return > median return (promising)
                0 otherwise (not promising)

        Uses a rolling window approach:
            For each date t:
                X = features from [t-lookback : t]
                y = 1 if return[t : t+forward] > median else 0

        Returns:
            X : np.ndarray — shape (n_samples, n_features)
            y : np.ndarray — shape (n_samples,) binary labels
        """
        print("[FeatureExtractor] Building training data...")

        tickers  = list(self.returns_df.columns)
        dates    = self.returns_df.index
        X_list, y_list = [], []

        # Rolling window — step every 5 days for efficiency
        step = 5
        for t in range(lookback_days, len(dates) - forward_days, step):
            window_features = self.features_df.iloc[t - lookback_days: t]
            forward_returns = self.returns_df.iloc[t: t + forward_days].sum()

            median_return = forward_returns.median()

            for ticker in tickers:
                cols = self._get_ticker_columns(ticker)
                if not cols:
                    continue

                ticker_data = window_features[cols].dropna()
                if ticker_data.empty:
                    continue

                mean_vals = ticker_data.mean().values
                std_vals  = ticker_data.std().fillna(0).values
                last_vals = ticker_data.iloc[-1].values
                row       = np.concatenate([mean_vals, std_vals, last_vals])

                label = 1 if forward_returns.get(ticker, 0) > median_return else 0

                X_list.append(row)
                y_list.append(label)

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"[FeatureExtractor] Training data: X={X.shape}, y={y.shape}")
        print(f"[FeatureExtractor] Positive labels: {y.sum()} / {len(y)}")
        return X, y
