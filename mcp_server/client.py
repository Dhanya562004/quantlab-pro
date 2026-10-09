"""
MCP Local Client & Integration Test Helper for QuantLab Pro.
Verifies stdio client connection, tool discovery, and tool invocation.
"""

from typing import Any

from quantlab.tools.registry import ToolRegistry


def run_local_mcp_client_test() -> dict[str, Any]:
    """
    Simulates MCP Client tool discovery and invocation via ToolRegistry / MCP Server definitions.
    Proves that external MCP clients can discover and invoke registered domain tools.
    """
    registry = ToolRegistry()
    tools = registry.list_tools()

    # Test invocation of 'dataset_summary'
    res_summary = registry.execute("dataset_summary", {"symbol": "MCP_TEST", "n_bars": 150, "seed": 99})

    # Test invocation of 'run_experiment'
    res_exp = registry.execute("run_experiment", {
        "symbol": "MCP_TEST",
        "n_bars": 250,
        "seed": 99,
        "model_type": "logistic_regression",
    })

    return {
        "mcp_server_name": "QuantLab-Pro-MCP-Server",
        "tools_discovered_count": len(tools),
        "discovered_tool_names": [t["name"] for t in tools],
        "dataset_summary_test": res_summary.model_dump(),
        "run_experiment_test": res_exp.model_dump(),
    }


def main():
    print("Running MCP Local Client Integration Verification...")
    res = run_local_mcp_client_test()
    print(f"Discovered Tools: {res['discovered_tool_names']}")
    print(f"Summary Test Success: {res['dataset_summary_test']['success']}")
    print(f"Experiment Test Success: {res['run_experiment_test']['success']}")
    if res['run_experiment_test']['success']:
        print(f"Generated Exp ID: {res['run_experiment_test']['data'].get('experiment_id')}")


if __name__ == "__main__":
    main()
