"""
Official Model Context Protocol (MCP) Server for QuantLab Pro.
Exposes allowlisted quantitative research tools via stdio transport.

Supported Tools:
- dataset_summary: Returns dataset shape, columns, and provenance fingerprint.
- validate_dataset: Runs data quality validation audit.
- run_experiment: Executes end-to-end ML training and backtesting experiment.
- get_experiment_metrics: Retrieves recorded metrics from SQLite history.
"""

import sys
from typing import Any

# Import MCP Server (Supports mcp 2.x MCPServer and mcp 1.x FastMCP)
try:
    from mcp.server.mcpserver import MCPServer as ServerClass
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as ServerClass
    except ImportError:
        from mcp.server import Server as ServerClass

from quantlab.tools.domain_tools import (
    DatasetSummaryArgs,
    GetExperimentArgs,
    RunExperimentArgs,
    ValidateDatasetArgs,
    tool_dataset_summary,
    tool_get_backtest_summary,
    tool_get_experiment,
    tool_get_experiment_metrics,
    tool_get_model_metrics,
    tool_inspect_dataset,
    tool_run_experiment,
    tool_validate_dataset,
)

# Instantiate MCP Server instance
mcp_server = ServerClass(
    name="QuantLab-Pro-MCP-Server",
    description="Quantitative ML Research Platform MCP Server exposing allowlisted research tools."
)


# --- Register Tools ---

@mcp_server.tool(
    name="dataset_summary",
    description="Generate a statistical summary and provenance fingerprint for a market dataset."
)
def dataset_summary(symbol: str = "SYNTH_BTC", n_bars: int = 300, seed: int = 42) -> dict[str, Any]:
    args = DatasetSummaryArgs(symbol=symbol, n_bars=n_bars, seed=seed)
    res = tool_dataset_summary(args)
    return res.model_dump()


@mcp_server.tool(
    name="validate_dataset",
    description="Run quantitative data quality validation checks."
)
def validate_dataset(symbol: str = "SYNTH_BTC", n_bars: int = 300, seed: int = 42) -> dict[str, Any]:
    args = ValidateDatasetArgs(symbol=symbol, n_bars=n_bars, seed=seed)
    report = tool_validate_dataset(args)
    return report.model_dump()


@mcp_server.tool(
    name="run_experiment",
    description="Execute end-to-end ML training and backtest experiment pipeline."
)
def run_experiment(
    symbol: str = "SYNTH_BTC",
    n_bars: int = 300,
    seed: int = 42,
    model_type: str = "logistic_regression",
    target_horizon: int = 1,
    transaction_cost_bps: float = 10.0
) -> dict[str, Any]:
    args = RunExperimentArgs(
        symbol=symbol,
        n_bars=n_bars,
        seed=seed,
        model_type=model_type,
        target_horizon=target_horizon,
        transaction_cost_bps=transaction_cost_bps,
    )
    return tool_run_experiment(args)


@mcp_server.tool(
    name="inspect_dataset",
    description="Inspect statistical dataset summary and provenance fingerprint (alias for dataset_summary)."
)
def inspect_dataset(symbol: str = "SYNTH_BTC", n_bars: int = 300, seed: int = 42) -> dict[str, Any]:
    args = DatasetSummaryArgs(symbol=symbol, n_bars=n_bars, seed=seed)
    res = tool_inspect_dataset(args)
    return res.model_dump()


@mcp_server.tool(
    name="get_experiment_metrics",
    description="Fetch recorded performance metrics for a specific experiment ID."
)
def get_experiment_metrics(experiment_id: str) -> dict[str, Any]:
    args = GetExperimentArgs(experiment_id=experiment_id)
    return tool_get_experiment_metrics(args)


@mcp_server.tool(
    name="get_experiment",
    description="Fetch full experiment details and manifest by experiment ID."
)
def get_experiment(experiment_id: str) -> dict[str, Any]:
    args = GetExperimentArgs(experiment_id=experiment_id)
    return tool_get_experiment(args)


@mcp_server.tool(
    name="get_model_metrics",
    description="Fetch model classification accuracy, precision, recall, and F1 score."
)
def get_model_metrics(experiment_id: str) -> dict[str, Any]:
    args = GetExperimentArgs(experiment_id=experiment_id)
    return tool_get_model_metrics(args)


@mcp_server.tool(
    name="get_backtest_summary",
    description="Fetch backtest Sharpe ratio and maximum drawdown for an experiment."
)
def get_backtest_summary(experiment_id: str) -> dict[str, Any]:
    args = GetExperimentArgs(experiment_id=experiment_id)
    return tool_get_backtest_summary(args)



def main():
    """Start MCP Server using stdio transport."""
    print("Starting QuantLab Pro MCP Server on stdio...", file=sys.stderr)
    mcp_server.run()


if __name__ == "__main__":
    main()
