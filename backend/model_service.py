import pandas as pd

from backend.models import ScreeningEngine, FEATURE_COLS


class ScreeningService:

    def __init__(self):
        self.engine = ScreeningEngine()

        self.is_trained = False

        self.metrics = {
            "accuracy": 0.0,
            "roc_auc": 0.0
        }


    # ============================================================
    # TRAIN MODEL
    # ============================================================

    def train(self, df: pd.DataFrame):

        self.engine.train(df)

        self.is_trained = True

        self.metrics = getattr(
            self.engine,
            "metrics",
            {
                "accuracy": 0.0,
                "roc_auc": 0.0
            }
        )


    # ============================================================
    # SINGLE COMPONENT PREDICTION
    # ============================================================

    def predict(self, payload: dict):

        row = pd.DataFrame([payload])

        missing = [
            column
            for column in FEATURE_COLS
            if column not in row.columns
        ]

        if missing:
            raise ValueError(
                "Missing prediction features: "
                + ", ".join(missing)
            )

        result = self.engine.predict_component(
            row[FEATURE_COLS]
        )

        if "shap_importance" in result:

            result["shap_importance"] = {
                str(key): float(value)
                for key, value
                in result["shap_importance"].items()
            }

        return result


    # ============================================================
    # BATCH PREDICTION
    # ============================================================

    def predict_batch(self, records):

        df = pd.DataFrame(records)

        missing = [
            column
            for column in FEATURE_COLS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "Missing prediction features: "
                + ", ".join(missing)
            )

        result = self.engine.predict_batch(
            df[FEATURE_COLS]
        )

        result = result.copy()

        result["Decision"] = (
            result["Decision"]
            .astype(str)
        )

        result["failure_probability"] = (
            result["failure_probability"]
            .astype(float)
        )

        result["anomaly_score"] = (
            result["anomaly_score"]
            .astype(float)
        )

        result["anomaly_flag"] = (
            result["anomaly_flag"]
            .astype(int)
        )

        return result.to_dict(
            orient="records"
        )