# Rubric: Task 03 - Repo Conventions

## Procedural Test Checks
1. `test_pydantic_state_schema`: Schema extension with `StressReport` Pydantic model and fields in `WorkflowState`.
2. `test_agent_inherits_and_runs`: Subclassing `BaseAgent`, `run()` method, safe execution without `backtest_result`, and trace logging.
3. `test_agent_exported_in_all`: Package exports and `__all__` list alignment in `quantlab/agents/__init__.py`.
4. `test_registry_integration`: Registration in allowlisted `ToolRegistry` with typed Pydantic args.
5. `test_orchestrator_pipeline_sequence`: Pipeline orchestration with 7 agents running in order.
6. `test_agent_tests_exist_and_pass`: Unit tests created in `tests/test_stress_agent.py` and passing.
7. `test_documentation_updated`: `docs/agents.md` accurately documents the 7-agent pipeline.

