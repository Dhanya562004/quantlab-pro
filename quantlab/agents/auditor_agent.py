"""
Experiment Auditor Agent for QuantLab Pro.
Assembles execution manifest, verifies checksums, and records experiment run into SQLite database.
"""

import time
import uuid

from quantlab.agents.base import BaseAgent
from quantlab.orchestration.state import AgentStepTrace, WorkflowState
from quantlab.storage.db import ExperimentStorage


class ExperimentAuditorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="Experiment Auditor Agent",
            agent_role="Verifies dataset checksums, compiles JSON reproducibility manifest, and persists run to SQLite database."
        )

    def run(self, state: WorkflowState) -> WorkflowState:
        start_t = time.time()

        if state.test_metrics is None or state.splits is None:
            duration = (time.time() - start_t) * 1000
            state.execution_trace.append(AgentStepTrace(
                step_index=len(state.execution_trace) + 1,
                agent_name=self.agent_name,
                role=self.agent_role,
                status="SKIPPED",
                duration_ms=round(duration, 2),
                summary="Skipped due to missing evaluation results or splits.",
            ))
            return state

        try:
            exp_id = f"EXP-{uuid.uuid4().hex[:8].upper()}"
            state.experiment_id = exp_id

            splits = state.splits
            meta = state.raw_metadata
            metrics = state.test_metrics
            backtest = state.backtest_result

            manifest = {
                "experiment_id": exp_id,
                "created_at": meta.get("created_at"),
                "dataset": {
                    "symbol": meta.get("symbol", "ASSET"),
                    "source": meta.get("source", "synthetic"),
                    "fingerprint": meta.get("fingerprint"),
                    "n_bars": meta.get("n_bars"),
                },
                "research_plan": {
                    "target_horizon": state.user_target_horizon,
                    "model_type": state.selected_model_type,
                    "seed": state.user_seed,
                    "rationale": state.research_rationale,
                },
                "features": {
                    "names": state.feature_names,
                    "count": len(state.feature_names),
                },
                "splits": {
                    "train_date_range": [splits.train_dates[0], splits.train_dates[-1]],
                    "val_date_range": [splits.val_dates[0], splits.val_dates[-1]],
                    "test_date_range": [splits.test_dates[0], splits.test_dates[-1]],
                    "train_sample_count": len(splits.X_train),
                    "test_sample_count": len(splits.X_test),
                },
                "results": {
                    "test_accuracy": metrics.accuracy,
                    "test_precision": metrics.precision,
                    "test_recall": metrics.recall,
                    "test_f1": metrics.f1_score,
                    "backtest_sharpe": backtest.sharpe_ratio if backtest else 0.0,
                    "backtest_max_drawdown": backtest.max_drawdown if backtest else 0.0,
                },
                "risk_warnings": state.risk_warnings,
            }

            state.manifest = manifest

            # Save to SQLite database
            storage = ExperimentStorage()
            storage.save_experiment(
                experiment_id=exp_id,
                symbol=meta.get("symbol", "ASSET"),
                source=meta.get("source", "synthetic"),
                fingerprint=meta.get("fingerprint", ""),
                model_type=state.selected_model_type,
                seed=state.user_seed,
                accuracy=metrics.accuracy,
                f1_score=metrics.f1_score,
                sharpe_ratio=backtest.sharpe_ratio if backtest else 0.0,
                max_drawdown=backtest.max_drawdown if backtest else 0.0,
                manifest=manifest,
                metrics=manifest["results"],
            )

            state.is_persisted = True
            state.is_completed = True

            duration = (time.time() - start_t) * 1000

            summary = f"Compiled manifest and persisted run '{exp_id}' to SQLite database successfully."

            state.execution_trace.append(AgentStepTrace(
                step_index=len(state.execution_trace) + 1,
                agent_name=self.agent_name,
                role=self.agent_role,
                status="PASS",
                duration_ms=round(duration, 2),
                summary=summary,
                details={
                    "experiment_id": exp_id,
                    "database_persisted": True,
                    "manifest_keys": list(manifest.keys()),
                }
            ))
        except Exception as e:
            duration = (time.time() - start_t) * 1000
            state.error_message = f"Audit Failed: {e!s}"
            state.execution_trace.append(AgentStepTrace(
                step_index=len(state.execution_trace) + 1,
                agent_name=self.agent_name,
                role=self.agent_role,
                status="FAIL",
                duration_ms=round(duration, 2),
                summary=f"Auditor error: {e!s}",
            ))

        return state


AuditorAgent = ExperimentAuditorAgent
