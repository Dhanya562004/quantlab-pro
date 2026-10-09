"""
Safe Allowlisted Tool Registry for QuantLab Pro.
Enforces Pydantic argument validation, execution limits, timing, and error handling.
"""

import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from quantlab.tools.domain_tools import (
    DatasetSummaryArgs,
    GetExperimentArgs,
    RunExperimentArgs,
    ValidateDatasetArgs,
    tool_dataset_summary,
    tool_get_experiment_manifest,
    tool_get_experiment_metrics,
    tool_run_experiment,
    tool_validate_dataset,
)


class ToolExecutionLog(BaseModel):
    """Log record for a executed tool call."""
    tool_name: str
    validated_args: dict[str, Any]
    success: bool
    execution_time_ms: float
    error: str | None = None


class ToolResult(BaseModel):
    """Result structure returned by tool registry execution."""
    success: bool
    data: Any | None = None
    error: str | None = None
    execution_time_ms: float = 0.0


class ToolDefinition(BaseModel):
    name: str
    description: str
    args_schema: type[BaseModel]
    func: Callable


class ToolRegistry:
    """Registry maintaining allowlisted tools with safe invocation."""
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self.execution_logs: list[ToolExecutionLog] = []
        self._register_default_tools()

    def register(self, name: str, description: str, args_schema: type[BaseModel], func: Callable):
        """Register an allowlisted tool with its Pydantic input schema."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            args_schema=args_schema,
            func=func
        )

    def _register_default_tools(self):
        self.register(
            name="dataset_summary",
            description="Generate a statistical summary and provenance fingerprint for a dataset.",
            args_schema=DatasetSummaryArgs,
            func=tool_dataset_summary
        )
        self.register(
            name="validate_dataset",
            description="Run quantitative data quality validation checks.",
            args_schema=ValidateDatasetArgs,
            func=tool_validate_dataset
        )
        self.register(
            name="run_experiment",
            description="Execute end-to-end ML training and backtest experiment pipeline.",
            args_schema=RunExperimentArgs,
            func=tool_run_experiment
        )
        self.register(
            name="get_experiment_metrics",
            description="Fetch recorded performance metrics for a specific experiment ID.",
            args_schema=GetExperimentArgs,
            func=tool_get_experiment_metrics
        )
        self.register(
            name="get_experiment_manifest",
            description="Retrieve full reproducibility manifest for a specific experiment ID.",
            args_schema=GetExperimentArgs,
            func=tool_get_experiment_manifest
        )

    def list_tools(self) -> list[dict[str, Any]]:
        """Return list of available allowlisted tools with their schemas."""
        output = []
        for name, tool in self._tools.items():
            output.append({
                "name": name,
                "description": tool.description,
                "parameters_schema": tool.args_schema.model_json_schema()
            })
        return output

    def execute(self, tool_name: str, raw_args: dict[str, Any]) -> ToolResult:
        """
        Validate input arguments against schema and safely invoke allowlisted tool.
        
        Args:
            tool_name: Registered tool identifier.
            raw_args: Dict of input argument values.
            
        Returns:
            ToolResult object.
        """
        start_t = time.time()
        
        if tool_name not in self._tools:
            duration_ms = round((time.time() - start_t) * 1000, 2)
            err_msg = f"Rejection: Tool '{tool_name}' is not in the allowlisted tool registry."
            self.execution_logs.append(ToolExecutionLog(
                tool_name=tool_name,
                validated_args=raw_args,
                success=False,
                execution_time_ms=duration_ms,
                error=err_msg
            ))
            return ToolResult(success=False, error=err_msg, execution_time_ms=duration_ms)

        tool = self._tools[tool_name]
        
        # Pydantic validation
        try:
            validated_pydantic_args = tool.args_schema(**raw_args)
        except Exception as ve:
            duration_ms = round((time.time() - start_t) * 1000, 2)
            err_msg = f"Argument validation error for tool '{tool_name}': {ve!s}"
            self.execution_logs.append(ToolExecutionLog(
                tool_name=tool_name,
                validated_args=raw_args,
                success=False,
                execution_time_ms=duration_ms,
                error=err_msg
            ))
            return ToolResult(success=False, error=err_msg, execution_time_ms=duration_ms)

        # Execute tool function
        try:
            res_data = tool.func(validated_pydantic_args)
            if isinstance(res_data, BaseModel):
                res_data = res_data.model_dump()
            duration_ms = round((time.time() - start_t) * 1000, 2)
            
            self.execution_logs.append(ToolExecutionLog(
                tool_name=tool_name,
                validated_args=validated_pydantic_args.model_dump(),
                success=True,
                execution_time_ms=duration_ms
            ))
            return ToolResult(success=True, data=res_data, execution_time_ms=duration_ms)
        except Exception as ex:
            duration_ms = round((time.time() - start_t) * 1000, 2)
            err_msg = f"Runtime error in tool '{tool_name}': {ex!s}"
            self.execution_logs.append(ToolExecutionLog(
                tool_name=tool_name,
                validated_args=validated_pydantic_args.model_dump(),
                success=False,
                execution_time_ms=duration_ms,
                error=err_msg
            ))
            return ToolResult(success=False, error=err_msg, execution_time_ms=duration_ms)
