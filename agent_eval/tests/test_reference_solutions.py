"""
Grader Self-Test: All reference solutions must pass 100% of grader tests.
"""

from pathlib import Path

import pytest

from agent_eval.runner.agents import DirectoryOverlayRunner
from agent_eval.runner.run_task import run_task

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"
TASK_NAMES = ["task_01_leakage", "task_02_ambiguous_spec", "task_03_conventions", "task_04_nondeterminism"]


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_reference_solution_passes(task_name: str):
    task_dir = TASKS_DIR / task_name
    ref_sol = task_dir / "reference_solution"
    assert ref_sol.exists(), f"Reference solution missing for {task_name}"

    runner = DirectoryOverlayRunner(ref_sol, name="reference_check")
    result = run_task(task_dir=task_dir, agent_or_solution=runner, timeout_seconds=45)

    assert result.passed is True, (
        f"Reference solution for {task_name} FAILED grader checks!\n"
        f"Tests: {result.tests_passed} passed, {result.tests_failed} failed\n"
        f"Grader Output:\n{result.test_output}"
    )
    assert result.tests_failed == 0
    assert result.tests_passed > 0
