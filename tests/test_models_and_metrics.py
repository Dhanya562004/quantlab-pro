"""
Unit Tests for Baseline Models, PyTorch Classifier, Metrics & Backtest (QuantLab Pro).
"""

import numpy as np

from quantlab.data.synthetic import generate_synthetic_ohlcv
from quantlab.evaluation.backtest import run_research_backtest
from quantlab.evaluation.metrics import compute_classification_metrics
from quantlab.features.builder import (
    FeatureConfig,
    build_features_and_target,
    create_chronological_splits,
)
from quantlab.models.baselines import (
    MajorityClassBaseline,
    SklearnLogisticRegressionModel,
)
from quantlab.models.pytorch_mlp import PyTorchMLPClassifier


def test_majority_class_baseline():
    X = np.random.randn(100, 5)
    y = np.array([1]*70 + [0]*30)
    
    model = MajorityClassBaseline()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(preds == 1)


def test_logistic_regression_model():
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = SklearnLogisticRegressionModel(seed=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 100
    assert set(preds).issubset({0, 1})


def test_pytorch_mlp_classifier():
    X = np.random.randn(80, 4)
    y = np.random.randint(0, 2, 80)
    
    model = PyTorchMLPClassifier(epochs=5, hidden_dim=16, seed=42)
    model.fit(X, y)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    
    assert len(preds) == 80
    assert probs.shape == (80, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_classification_metrics_computation():
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics.accuracy == 0.75
    assert len(metrics.confusion_matrix) == 2


def test_research_backtest_calculation():
    df, _ = generate_synthetic_ohlcv(symbol="TEST", n_bars=200, seed=42)
    config = FeatureConfig()
    X, y, feature_cols, df_aligned = build_features_and_target(df, config)
    splits = create_chronological_splits(X, y, df_aligned, config)
    
    preds = np.ones(len(splits.y_test), dtype=int)
    res = run_research_backtest(df_aligned, preds, splits.test_dates, transaction_cost_bps=10.0)
    
    assert len(res.strategy_curve) == len(splits.test_dates)
    assert res.num_trades >= 0
    assert res.transaction_cost_bps == 10.0
