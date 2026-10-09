"""
SQLite Experiment Database Storage Module for QuantLab Pro.
Provides persistent storage, retrieval, comparison, and resetting of experiment runs.
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

import numpy as np
from pydantic import BaseModel


class ExperimentRecord(BaseModel):
    """Database record representation of an executed experiment."""
    experiment_id: str
    created_at: str
    symbol: str
    source: str
    fingerprint: str
    model_type: str
    seed: int
    accuracy: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    manifest_json: dict[str, Any]
    metrics_json: dict[str, Any]


class ExperimentStorage:
    """SQLite Experiment Database Manager."""
    def __init__(self, db_path: str = "quantlab_experiments.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize SQLite table schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    source TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    accuracy REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    manifest_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL
                )
            """)
            conn.commit()

    def save_experiment(
        self,
        experiment_id: str,
        symbol: str,
        source: str,
        fingerprint: str,
        model_type: str,
        seed: int,
        accuracy: float,
        f1_score: float,
        sharpe_ratio: float,
        max_drawdown: float,
        manifest: dict[str, Any],
        metrics: dict[str, Any],
    ) -> bool:
        """Save a new experiment run to SQLite using parameterized SQL."""
        created_at = datetime.now(timezone.utc).isoformat()
        manifest_str = json.dumps(manifest, default=str)
        metrics_str = json.dumps(metrics, default=str)

        def _clean_float(val: Any) -> float:
            try:
                f_val = float(val)
                return 0.0 if (np.isnan(f_val) or np.isinf(f_val)) else f_val
            except Exception:
                return 0.0

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    experiment_id, created_at, symbol, source, fingerprint,
                    model_type, seed, accuracy, f1_score, sharpe_ratio, max_drawdown,
                    manifest_json, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    created_at,
                    symbol,
                    source,
                    fingerprint,
                    model_type,
                    seed,
                    _clean_float(accuracy),
                    _clean_float(f1_score),
                    _clean_float(sharpe_ratio),
                    _clean_float(max_drawdown),
                    manifest_str,
                    metrics_str,
                )
            )
            conn.commit()
        return True

    def list_experiments(self, limit: int = 50) -> list[dict[str, Any]]:
        """List past experiment runs ordered by creation time descending."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT experiment_id, created_at, symbol, source, model_type, seed, accuracy, f1_score, sharpe_ratio, max_drawdown FROM experiments ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        """Retrieve full experiment details and manifest by experiment_id."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,))
            row = cursor.fetchone()
            if not row:
                return None

            d = dict(row)
            return ExperimentRecord(
                experiment_id=d["experiment_id"],
                created_at=d["created_at"],
                symbol=d["symbol"],
                source=d["source"],
                fingerprint=d["fingerprint"],
                model_type=d["model_type"],
                seed=d["seed"],
                accuracy=d["accuracy"],
                f1_score=d["f1_score"],
                sharpe_ratio=d["sharpe_ratio"],
                max_drawdown=d["max_drawdown"],
                manifest_json=json.loads(d["manifest_json"]),
                metrics_json=json.loads(d["metrics_json"]),
            )

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete a single experiment record."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
            conn.commit()
        return True

    def clear_all_experiments(self) -> bool:
        """Clear all experiment records from the database (Reset History)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM experiments")
            conn.commit()
        return True
