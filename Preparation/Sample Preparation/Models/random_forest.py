"""Random Forest classifier wrapper"""
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class RandomForestModel:
    def __init__(self, n_estimators: int = 100, **kwargs):
        self.model = RandomForestClassifier(n_estimators=n_estimators, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
