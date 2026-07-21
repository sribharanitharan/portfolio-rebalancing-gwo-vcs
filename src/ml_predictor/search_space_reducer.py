import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')

from ml_predictor.feature_extractor import FeatureExtractor
from ml_predictor.region_predictor import RegionPredictor


class SearchSpaceReducer:

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        region_predictor: RegionPredictor,
        top_k: int = 10,
        min_k: int = 5
    ):
        self.extractor       = feature_extractor
        self.predictor       = region_predictor
        self.top_k           = top_k
        self.min_k           = min_k

        self.asset_scores    = None
        self.promising_idx   = None
        self.promising_names = None
        self.all_assets      = None

    def run(self, lookback_days: int = 60):
        print("[SearchSpaceReducer] Running ML space reduction...")

        feature_matrix  = self.extractor.build_asset_feature_matrix(
            lookback_days=lookback_days
        )
        self.all_assets = list(feature_matrix.index)
        n_assets        = len(self.all_assets)

        scores           = self.predictor.predict_scores(feature_matrix.values)
        self.asset_scores = pd.Series(scores, index=self.all_assets)

        k                    = max(self.min_k, min(self.top_k, n_assets))
        top_assets           = self.asset_scores.nlargest(k)
        self.promising_names = list(top_assets.index)
        self.promising_idx   = [
            self.all_assets.index(name) for name in self.promising_names
        ]

        print(f"[SearchSpaceReducer] Assets scored: {n_assets}")
        print(f"[SearchSpaceReducer] Top-{k} promising assets selected:")
        for name, score in top_assets.items():
            print(f"    {name:8s} score: {score:.4f}")

        return self.promising_idx

    def get_promising_indices(self) -> list:
        if self.promising_idx is None:
            self.run()
        return self.promising_idx

    def get_scores_dataframe(self) -> pd.DataFrame:
        if self.asset_scores is None:
            raise RuntimeError("Run .run() first to compute scores.")

        df = pd.DataFrame({
            "asset"    : self.asset_scores.index,
            "score"    : self.asset_scores.values,
            "promising": [
                i in self.promising_idx
                for i in range(len(self.asset_scores))
            ]
        })
        return df.sort_values("score", ascending=False).reset_index(drop=True)

    def update(self, lookback_days: int = 60):
        print("[SearchSpaceReducer] Updating asset scores...")
        self.promising_idx   = None
        self.promising_names = None
        return self.run(lookback_days=lookback_days)


def build_reducer(
    features_path: str = "data/processed/features.csv",
    returns_path: str  = "data/processed/returns.csv",
    model_type: str    = "rf",
    top_k: int         = 10,
    train_new: bool    = True
) -> SearchSpaceReducer:

    extractor = FeatureExtractor(features_path, returns_path).load()
    predictor = RegionPredictor(model_type=model_type)

    if train_new:
        X, y = extractor.build_training_data()
        predictor.train(X, y)
    else:
        predictor.load()

    reducer = SearchSpaceReducer(
        feature_extractor = extractor,
        region_predictor  = predictor,
        top_k             = top_k
    )

    return reducer
