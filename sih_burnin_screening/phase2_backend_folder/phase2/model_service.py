import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import ScreeningEngine, FEATURE_COLS


class ScreeningService:
    def __init__(self):
        self.engine = ScreeningEngine()
        self.is_trained = False
        self.metrics = {
            "accuracy": 0.0,
            "roc_auc": 0.0
        }

    def train(self, df: pd.DataFrame):
        self.engine.train(df)
        self.is_trained = True

        # Uses metrics if the existing ML engine provides them.
        self.metrics = getattr(
            self.engine,
            "metrics",
            {
                "accuracy": 0.0,
                "roc_auc": 0.0
            }
        )

    def predict(self, payload: dict):
        row = pd.DataFrame([payload])

        missing = [
            column for column in FEATURE_COLS
            if column not in row.columns
        ]

        if missing:
            raise ValueError(
                "Missing prediction features: " + ", ".join(missing)
            )

        result = self.engine.predict_component(
            row[FEATURE_COLS]
        )

        if "shap_importance" in result:
            result["shap_importance"] = {
                str(key): float(value)
                for key, value in result["shap_importance"].items()
            }

        return result
