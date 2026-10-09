"""
Unit & Integration Tests for MCP Server Tool Discovery and Local Client (QuantLab Pro).
"""


from mcp_server.client import run_local_mcp_client_test


def test_mcp_local_client_tool_discovery_and_invocation():
    res = run_local_mcp_client_test()

    assert res["mcp_server_name"] == "QuantLab-Pro-MCP-Server"
    assert res["tools_discovered_count"] >= 4
    assert "dataset_summary" in res["discovered_tool_names"]
    assert "run_experiment" in res["discovered_tool_names"]

    assert res["dataset_summary_test"]["success"] is True
    assert res["run_experiment_test"]["success"] is True
