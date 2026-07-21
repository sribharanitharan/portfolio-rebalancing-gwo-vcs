import numpy as np
import pandas as pd
import pickle
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


class RegionPredictor:

    def __init__(
        self,
        model_type: str  = "rf",
        model_path: str  = "models/trained_predictor.pkl",
        scaler_path: str = "models/scaler.pkl"
    ):
        self.model_type  = model_type
        self.model_path  = model_path
        self.scaler_path = scaler_path
        self.model       = None
        self.scaler      = StandardScaler()
        self.is_trained  = False

        os.makedirs("models", exist_ok=True)

    def _build_model(self):
        if self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators = 100,
                max_depth    = 8,
                random_state = 42,
                n_jobs       = -1
            )
        elif self.model_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes  = (128, 64, 32),
                activation          = "relu",
                max_iter            = 300,
                random_state        = 42,
                early_stopping      = True,
                validation_fraction = 0.1
            )
        elif self.model_type == "gb":
            return GradientBoostingClassifier(
                n_estimators  = 100,
                learning_rate = 0.05,
                max_depth     = 5,
                random_state  = 42
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. "
                             f"Choose from 'rf', 'mlp', 'gb'")

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        print(f"[RegionPredictor] Training {self.model_type.upper()} model...")
        print(f"  Input shape : X={X.shape}, y={y.shape}")

        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)

        self.model = self._build_model()
        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"\n[RegionPredictor] Results:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"\n{classification_report(y_test, y_pred)}")

        self.save()
        return {"accuracy": acc, "auc": auc}

    def predict_scores(self, feature_matrix: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("[RegionPredictor] Model not trained yet. "
                               "Call train() or load() first.")

        X = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        X = self.scaler.transform(X)

        scores = self.model.predict_proba(X)[:, 1]
        return scores

    def save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"[RegionPredictor] Model saved to: {self.model_path}")

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[RegionPredictor] No saved model at {self.model_path}. "
                f"Train first."
            )
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.is_trained = True
        print(f"[RegionPredictor] Model loaded from: {self.model_path}")
        return self
