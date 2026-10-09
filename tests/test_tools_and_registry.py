"""
Unit Tests for Safe Tool-Calling Registry (QuantLab Pro).
Verifies Pydantic argument validation, execution logging, and unregistered tool rejection.
"""


from quantlab.tools.registry import ToolRegistry


def test_tool_registry_discovery():
    registry = ToolRegistry()
    tools = registry.list_tools()
    tool_names = [t["name"] for t in tools]

    assert "dataset_summary" in tool_names
    assert "validate_dataset" in tool_names
    assert "run_experiment" in tool_names
    assert "get_experiment_metrics" in tool_names
    assert "get_experiment_manifest" in tool_names


def test_tool_execution_valid_args():
    registry = ToolRegistry()
    res = registry.execute("dataset_summary", {"symbol": "TOOL_TEST", "n_bars": 150, "seed": 42})

    assert res.success is True
    assert res.data["symbol"] == "TOOL_TEST"
    assert res.data["n_bars"] == 150
    assert "fingerprint" in res.data


def test_tool_execution_invalid_args_schema():
    registry = ToolRegistry()
    # n_bars expects integer, passing unparseable string
    res = registry.execute("dataset_summary", {"symbol": "TOOL_TEST", "n_bars": "invalid_number"})

    assert res.success is False
    assert "validation error" in res.error.lower()


def test_unregistered_tool_rejection():
    registry = ToolRegistry()
    res = registry.execute("arbitrary_shell_exec", {"command": "rm -rf /"})

    assert res.success is False
    assert "allowlisted" in res.error.lower()
