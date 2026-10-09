"""
Domain Tool Implementations for QuantLab Pro.
Reusable, typed functions exposed to both Multi-Agent Orchestrator and MCP Server.
"""

from typing import Any

from pydantic import BaseModel, Field

from quantlab.data.synthetic import generate_synthetic_ohlcv
from quantlab.data.validation import ValidationReport, validate_ohlcv_data
from quantlab.evaluation.backtest import run_research_backtest
from quantlab.evaluation.metrics import compute_classification_metrics
from quantlab.features.builder import (
    FeatureConfig,
    build_features_and_target,
    create_chronological_splits,
)
from quantlab.models.trainer import train_model
from quantlab.storage.db import ExperimentStorage

# --- Pydantic Schemas for Tool Arguments & Results ---

class DatasetSummaryArgs(BaseModel):
    symbol: str = Field(default="SYNTH_BTC", description="Asset symbol.")
    n_bars: int = Field(default=300, description="Number of bars.")
    seed: int = Field(default=42, description="Random seed.")


class DatasetSummaryResult(BaseModel):
    symbol: str
    n_bars: int
    columns: list[str]
    start_date: str
    end_date: str
    fingerprint: str
    provenance: str


class ValidateDatasetArgs(BaseModel):
    symbol: str = Field(default="SYNTH_BTC")
    n_bars: int = Field(default=300)
    seed: int = Field(default=42)


class RunExperimentArgs(BaseModel):
    symbol: str = Field(default="SYNTH_BTC", description="Asset symbol.")
    n_bars: int = Field(default=300, description="Number of bars (100 to 5000).")
    seed: int = Field(default=42, description="Random seed for data & model.")
    model_type: str = Field(default="logistic_regression", description="Model architecture ('majority_class', 'logistic_regression', 'pytorch_mlp').")
    target_horizon: int = Field(default=1, description="Forecast horizon in bars.")
    transaction_cost_bps: float = Field(default=10.0, description="Backtest transaction cost in bps.")


class GetExperimentArgs(BaseModel):
    experiment_id: str = Field(description="Unique experiment UUID/ID string.")


# --- Implementations ---

def tool_dataset_summary(args: DatasetSummaryArgs) -> DatasetSummaryResult:
    df, meta = generate_synthetic_ohlcv(symbol=args.symbol, n_bars=args.n_bars, seed=args.seed)
    return DatasetSummaryResult(
        symbol=meta["symbol"],
        n_bars=meta["n_bars"],
        columns=list(df.columns),
        start_date=meta["start_date"],
        end_date=meta["end_date"],
        fingerprint=meta["fingerprint"],
        provenance=meta["source"],
    )


def tool_validate_dataset(args: ValidateDatasetArgs) -> ValidationReport:
    df, _ = generate_synthetic_ohlcv(symbol=args.symbol, n_bars=args.n_bars, seed=args.seed)
    return validate_ohlcv_data(df)


def tool_run_experiment(args: RunExperimentArgs) -> dict[str, Any]:
    # 1. Generate Data
    df, meta = generate_synthetic_ohlcv(symbol=args.symbol, n_bars=args.n_bars, seed=args.seed)

    # 2. Validate Data
    val_report = validate_ohlcv_data(df)
    if not val_report.is_valid:
        return {"success": False, "error": f"Validation failed: {val_report.errors}"}

    # 3. Features & Splits
    config = FeatureConfig(target_horizon=args.target_horizon)
    X, y, feat_cols, df_aligned = build_features_and_target(df, config)
    splits = create_chronological_splits(X, y, df_aligned, config)

    # 4. Train Model
    artifact, model = train_model(args.model_type, splits, seed=args.seed)

    # 5. Evaluate held-out Test split
    y_test_pred = model.predict(splits.X_test)
    test_metrics = compute_classification_metrics(splits.y_test, y_test_pred)

    # 6. Backtest
    backtest_res = run_research_backtest(df_aligned, y_test_pred, splits.test_dates, transaction_cost_bps=args.transaction_cost_bps)

    # 7. Persist to DB
    import uuid
    exp_id = f"EXP-{uuid.uuid4().hex[:8].upper()}"
    storage = ExperimentStorage()

    manifest = {
        "experiment_id": exp_id,
        "symbol": args.symbol,
        "source": meta["source"],
        "fingerprint": meta["fingerprint"],
        "model_type": args.model_type,
        "seed": args.seed,
        "feature_config": config.model_dump(),
        "train_dates": [splits.train_dates[0], splits.train_dates[-1]],
        "test_dates": [splits.test_dates[0], splits.test_dates[-1]],
    }

    metrics_summary = {
        "accuracy": test_metrics.accuracy,
        "precision": test_metrics.precision,
        "recall": test_metrics.recall,
        "f1_score": test_metrics.f1_score,
        "sharpe_ratio": backtest_res.sharpe_ratio,
        "max_drawdown": backtest_res.max_drawdown,
    }

    storage.save_experiment(
        experiment_id=exp_id,
        symbol=args.symbol,
        source=meta["source"],
        fingerprint=meta["fingerprint"],
        model_type=args.model_type,
        seed=args.seed,
        accuracy=test_metrics.accuracy,
        f1_score=test_metrics.f1_score,
        sharpe_ratio=backtest_res.sharpe_ratio,
        max_drawdown=backtest_res.max_drawdown,
        manifest=manifest,
        metrics=metrics_summary,
    )

    return {
        "success": True,
        "experiment_id": exp_id,
        "metrics": metrics_summary,
        "backtest": backtest_res.model_dump(),
        "manifest": manifest,
    }


def tool_get_experiment_metrics(args: GetExperimentArgs) -> dict[str, Any]:
    storage = ExperimentStorage()
    rec = storage.get_experiment(args.experiment_id)
    if not rec:
        return {"success": False, "error": f"Experiment '{args.experiment_id}' not found."}
    return {"success": True, "experiment_id": rec.experiment_id, "metrics": rec.metrics_json}


def tool_get_experiment_manifest(args: GetExperimentArgs) -> dict[str, Any]:
    storage = ExperimentStorage()
    rec = storage.get_experiment(args.experiment_id)
    if not rec:
        return {"success": False, "error": f"Experiment '{args.experiment_id}' not found."}
    return {"success": True, "experiment_id": rec.experiment_id, "manifest": rec.manifest_json}
