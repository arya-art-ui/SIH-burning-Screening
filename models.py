import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import shap


FEATURE_COLS = [
    "v_0h",
    "i_0h",
    "temp_0h",
    "v_24h",
    "i_24h",
    "temp_24h",
    "delta_v_0_24",
    "delta_i_0_24",
    "delta_temp_0_24",
]


class ScreeningEngine:
    """Anomaly detection + 168h failure prediction + SHAP explanation."""

    def __init__(self):
        self.iso_forest = IsolationForest(
            contamination=0.10,
            random_state=42,
        )
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
        self.explainer = None
        self.is_trained = False
        self.metrics = {}

    def train(self, df: pd.DataFrame):
        X = df[FEATURE_COLS].copy()
        y = df["actual_failure_168h"].astype(int)

        self.iso_forest.fit(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        self.xgb_model.fit(X_train, y_train)

        predictions = self.xgb_model.predict(X_test)
        probabilities = self.xgb_model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }

        self.explainer = shap.TreeExplainer(self.xgb_model)
        self.is_trained = True

    @staticmethod
    def _decision_from_arrays(failure_probability, anomaly_flag, anomaly_score):
        failure_probability = np.asarray(failure_probability)
        anomaly_flag = np.asarray(anomaly_flag)
        anomaly_score = np.asarray(anomaly_score)

        decision = np.full(len(failure_probability), "PASS", dtype=object)

        warning_mask = (
            (failure_probability > 0.25)
            | (anomaly_score < -0.40)
        )
        reject_mask = (
            (failure_probability > 0.60)
            | (anomaly_flag == -1)
        )

        decision[warning_mask] = "WARNING"
        decision[reject_mask] = "REJECT"
        return decision

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fast batch prediction used by dashboard tables and KPIs.

        SHAP is intentionally not calculated here. SHAP is expensive and is
        only needed when the user opens an individual AI explanation.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        X = df[FEATURE_COLS].copy()

        anomaly_flag = self.iso_forest.predict(X)
        anomaly_score = self.iso_forest.score_samples(X)
        failure_probability = self.xgb_model.predict_proba(X)[:, 1]

        decisions = self._decision_from_arrays(
            failure_probability,
            anomaly_flag,
            anomaly_score,
        )

        return pd.DataFrame({
            "failure_probability": failure_probability.astype(float),
            "anomaly_score": anomaly_score.astype(float),
            "anomaly_flag": anomaly_flag.astype(int),
            "Decision": decisions,
        }, index=df.index)

    def predict_component(self, X_sample: pd.DataFrame):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        X_sample = X_sample[FEATURE_COLS].copy()

        anomaly_flag = int(self.iso_forest.predict(X_sample)[0])
        raw_anomaly_score = float(self.iso_forest.score_samples(X_sample)[0])
        anomaly_score = float(np.clip(raw_anomaly_score, -1.0, 1.0))

        failure_prob = float(
            self.xgb_model.predict_proba(X_sample)[0][1]
        )

        if failure_prob > 0.60 or anomaly_flag == -1:
            decision = "REJECT"
            risk_level = "HIGH RISK"
        elif failure_prob > 0.25 or anomaly_score < -0.40:
            decision = "WARNING"
            risk_level = "MEDIUM RISK"
        else:
            decision = "PASS"
            risk_level = "LOW RISK"

        shap_output = self.explainer(X_sample)
        shap_values = np.asarray(shap_output.values)

        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        shap_row = shap_values[0]

        shap_importance = {
            feature: float(value)
            for feature, value in zip(FEATURE_COLS, shap_row)
        }

        return {
            "decision": decision,
            "risk_level": risk_level,
            "failure_probability": failure_prob,
            "anomaly_score": anomaly_score,
            "anomaly_flag": anomaly_flag,
            "shap_importance": shap_importance,
        }
