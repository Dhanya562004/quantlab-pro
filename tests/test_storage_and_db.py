"""
Unit Tests for SQLite Experiment Storage & Persistence (QuantLab Pro).
"""

import os

import pytest

from quantlab.storage.db import ExperimentStorage


@pytest.fixture
def temp_db(tmp_path):
    db_file = os.path.join(tmp_path, "test_experiments.db")
    storage = ExperimentStorage(db_path=db_file)
    yield storage


def test_sqlite_save_and_retrieve(temp_db):
    exp_id = "EXP-TEST-100"
    manifest = {"exp_id": exp_id, "test": True}
    metrics = {"accuracy": 0.65, "f1_score": 0.62}
    
    saved = temp_db.save_experiment(
        experiment_id=exp_id,
        symbol="BTC",
        source="synthetic",
        fingerprint="abc123hash",
        model_type="logistic_regression",
        seed=42,
        accuracy=0.65,
        f1_score=0.62,
        sharpe_ratio=1.2,
        max_drawdown=-0.15,
        manifest=manifest,
        metrics=metrics,
    )
    assert saved is True
    
    record = temp_db.get_experiment(exp_id)
    assert record is not None
    assert record.experiment_id == exp_id
    assert record.accuracy == 0.65
    assert record.manifest_json == manifest


def test_sqlite_clear_history(temp_db):
    temp_db.save_experiment(
        experiment_id="EXP-1", symbol="BTC", source="synthetic", fingerprint="f1",
        model_type="majority_class", seed=1, accuracy=0.5, f1_score=0.5,
        sharpe_ratio=0.0, max_drawdown=0.0, manifest={}, metrics={}
    )
    assert len(temp_db.list_experiments()) == 1
    
    temp_db.clear_all_experiments()
    assert len(temp_db.list_experiments()) == 0
