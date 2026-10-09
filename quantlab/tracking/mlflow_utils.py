"""
MLflow Optional Experiment Tracking Integration for QuantLab Pro.
Fails safely if MLflow server or package is unavailable.
"""

from typing import Any


def log_experiment_to_mlflow(
    experiment_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    tags: dict[str, str] | None = None,
) -> bool:
    """
    Attempt to log experiment parameters and metrics to an MLflow tracking server.

    If MLflow is not installed or the server connection fails, returns False without raising exceptions.

    Args:
        experiment_name: MLflow experiment name.
        params: Key-value parameters.
        metrics: Key-value numeric evaluation metrics.
        tags: Optional metadata tags.

    Returns:
        True if successfully logged to MLflow, False otherwise.
    """
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(tags=tags):
            # Log params
            for k, v in params.items():
                mlflow.log_param(k, str(v))

            # Log metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))

        return True
    except Exception:
        # Silently degrade: MLflow is an optional secondary tracking server
        return False
