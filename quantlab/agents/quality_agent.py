"""
Data Quality Agent for QuantLab Pro.
Validates schema, price integrity, date monotonicity, missing values, and dataset provenance.
"""

import time

from quantlab.agents.base import BaseAgent
from quantlab.data.validation import validate_ohlcv_data
from quantlab.orchestration.state import AgentStepTrace, WorkflowState


class DataQualityAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="Data Quality Agent",
            agent_role="Validates dataset schema, pricing integrity, temporal ordering, and sample adequacy."
        )

    def run(self, state: WorkflowState) -> WorkflowState:
        start_t = time.time()

        if state.df_raw is None or len(state.df_raw) == 0:
            duration = (time.time() - start_t) * 1000
            state.error_message = "Data Quality Agent Error: No dataset provided in state."
            state.execution_trace.append(AgentStepTrace(
                step_index=len(state.execution_trace) + 1,
                agent_name=self.agent_name,
                role=self.agent_role,
                status="FAIL",
                duration_ms=round(duration, 2),
                summary="No raw data present in workflow state.",
            ))
            return state

        report = validate_ohlcv_data(state.df_raw)
        state.validation_report = report

        duration = (time.time() - start_t) * 1000

        if not report.is_valid:
            state.error_message = f"Data Quality Validation Failed: {report.errors}"
            status = "FAIL"
            summary = f"Validation failed with {len(report.errors)} blocking errors."
        else:
            status = "PASS"
            summary = f"Data quality validation passed successfully ({report.metrics.get('total_rows', 0)} bars verified)."

        state.execution_trace.append(AgentStepTrace(
            step_index=len(state.execution_trace) + 1,
            agent_name=self.agent_name,
            role=self.agent_role,
            status=status,
            duration_ms=round(duration, 2),
            summary=summary,
            details={
                "errors": report.errors,
                "warnings": report.warnings,
                "metrics": report.metrics,
                "audit_steps": len(report.audit_trail),
            }
        ))

        return state
