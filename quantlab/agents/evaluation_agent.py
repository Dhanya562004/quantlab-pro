"""
Evaluation & Risk Agent for QuantLab Pro.
Evaluates model on held-out test split, computes backtest, and checks for data leakage warnings.
"""

import time

import numpy as np

from quantlab.agents.base import BaseAgent
from quantlab.evaluation.backtest import run_research_backtest
from quantlab.evaluation.metrics import compute_classification_metrics
from quantlab.orchestration.state import AgentStepTrace, WorkflowState


class EvaluationRiskAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="Evaluation & Risk Agent",
            agent_role="Evaluates held-out test accuracy, precision/recall, backtest returns, and checks for leakage flags."
        )

    def run(self, state: WorkflowState) -> WorkflowState:
        start_t = time.time()
        
        if state.trained_model_object is None or state.splits is None:
            duration = (time.time() - start_t) * 1000
            state.execution_trace.append(AgentStepTrace(
                step_index=len(state.execution_trace) + 1,
                agent_name=self.agent_name,
                role=self.agent_role,
                status="SKIPPED",
                duration_ms=round(duration, 2),
                summary="Skipped due to missing trained model or splits.",
            ))
            return state

        try:
            model = state.trained_model_object
            splits = state.splits
            
            # Predict on held-out test split
            y_test_pred = model.predict(splits.X_test)
            test_metrics = compute_classification_metrics(splits.y_test, y_test_pred)
            
            # Run research backtest
            backtest_res = run_research_backtest(
                df_aligned=splits.df_aligned,
                predictions=y_test_pred,
                test_dates=splits.test_dates,
                transaction_cost_bps=state.user_transaction_cost_bps,
            )
            
            state.test_metrics = test_metrics
            state.backtest_result = backtest_res
            
            # Risk & Leakage Auditing Flags
            risk_warnings = []
            
            # 1. Suspicious Accuracy Check (> 90% in financial time series is usually suspicious leakage)
            if test_metrics.accuracy > 0.90:
                risk_warnings.append(
                    f"SUSPICIOUS HIGH ACCURACY ({test_metrics.accuracy * 100:.1f}%). "
                    "Verify features do not contain future leakage or look-ahead bias."
                )
                
            # 2. Class Imbalance Check
            train_pos_pct = np.mean(splits.y_train == 1)
            if train_pos_pct < 0.35 or train_pos_pct > 0.65:
                risk_warnings.append(
                    f"Class Imbalance Warning: Training target distribution is skewed ({train_pos_pct*100:.1f}% positive)."
                )
                
            # 3. Small Test Sample Warning
            if len(splits.y_test) < 50:
                risk_warnings.append(
                    f"Small Held-Out Test Set ({len(splits.y_test)} bars). Statistical confidence intervals are wide."
                )
                
            state.risk_warnings = risk_warnings
            
            duration = (time.time() - start_t) * 1000
            
            summary = (
                f"Evaluated held-out Test split ({len(splits.y_test)} bars). "
                f"Accuracy={test_metrics.accuracy * 100:.1f}%, F1={test_metrics.f1_score:.2f}, "
                f"Backtest Sharpe={backtest_res.sharpe_ratio:.2f}, Max Drawdown={backtest_res.max_drawdown * 100:.1f}%."
            )
            
            state.execution_trace.append(AgentStepTrace(
                step_index=len(state.execution_trace) + 1,
                agent_name=self.agent_name,
                role=self.agent_role,
                status="PASS",
                duration_ms=round(duration, 2),
                summary=summary,
                details={
                    "test_accuracy": test_metrics.accuracy,
                    "test_f1": test_metrics.f1_score,
                    "confusion_matrix": test_metrics.confusion_matrix,
                    "sharpe_ratio": backtest_res.sharpe_ratio,
                    "max_drawdown": backtest_res.max_drawdown,
                    "num_trades": backtest_res.num_trades,
                    "risk_warnings": risk_warnings,
                }
            ))
        except Exception as e:
            duration = (time.time() - start_t) * 1000
            state.error_message = f"Evaluation Failed: {e!s}"
            state.execution_trace.append(AgentStepTrace(
                step_index=len(state.execution_trace) + 1,
                agent_name=self.agent_name,
                role=self.agent_role,
                status="FAIL",
                duration_ms=round(duration, 2),
                summary=f"Evaluation error: {e!s}",
            ))
            
        return state
