"""
Baseline Machine Learning Models for QuantLab Pro.
Includes Majority Class Baseline and scikit-learn Logistic Regression.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class MajorityClassBaseline:
    """Baseline model that always predicts the majority class observed in training data."""
    def __init__(self):
        self.majority_class: int = 0
        self.class_probabilities: np.ndarray = np.array([0.5, 0.5])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityClassBaseline":
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) == 0:
            self.majority_class = 0
            self.class_probabilities = np.array([0.5, 0.5])
        else:
            self.majority_class = int(classes[np.argmax(counts)])
            p1 = np.mean(y == 1)
            self.class_probabilities = np.array([1 - p1, p1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(shape=(len(X),), fill_value=self.majority_class, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.class_probabilities, (len(X), 1))


class SklearnLogisticRegressionModel:
    """Scikit-learn Logistic Regression wrapper with random seed control."""
    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnLogisticRegressionModel":
        if len(np.unique(y)) < 2:
            # Single class fallback
            self.is_fitted = False
            self.majority_class = int(y[0]) if len(y) > 0 else 0
        else:
            self.model.fit(X, y)
            self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.full(shape=(len(X),), fill_value=getattr(self, "majority_class", 0), dtype=int)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            p = 1.0 if getattr(self, "majority_class", 0) == 1 else 0.0
            return np.tile(np.array([1 - p, p]), (len(X), 1))
        return self.model.predict_proba(X)
