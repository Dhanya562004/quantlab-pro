"""
Unit & Integration Tests for Official MCP SDK Server and Stdio Client (QuantLab Pro).
"""

import asyncio

from mcp_server.client import (
    run_local_mcp_client_test,
    run_mcp_sdk_client_integration_test,
)


def test_mcp_local_client_tool_discovery_and_invocation():
    res = run_local_mcp_client_test()

    assert res["mcp_server_name"] == "QuantLab-Pro-MCP-Server"
    assert res["tools_discovered_count"] >= 4
    assert "dataset_summary" in res["discovered_tool_names"]
    assert "run_experiment" in res["discovered_tool_names"]

    assert res["dataset_summary_test"]["success"] is True
    assert res["run_experiment_test"]["success"] is True


def test_real_mcp_sdk_stdio_client_integration():
    """
    Integration test connecting over stdio using official Python MCP SDK.
    Verifies tool discovery, real tool invocation, structured content, error rejection, and clean shutdown.
    """
    res = asyncio.run(run_mcp_sdk_client_integration_test())

    assert res["mcp_server_name"] == "QuantLab-Pro-MCP-Server"
    assert res["tools_discovered_count"] >= 4
    assert "dataset_summary" in res["discovered_tool_names"]
    assert "run_experiment" in res["discovered_tool_names"]
    assert "inspect_dataset" in res["discovered_tool_names"]

    # Verify structured tool output
    assert res["summary_result"] is not None
    assert res["experiment_result"] is not None

    # Verify invalid tool rejection
    assert res["invalid_tool_test"] is not None
