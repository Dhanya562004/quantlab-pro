"""
Grader Self-Test: Determinism check (runs grader N times on reference solution).
"""

from pathlib import Path

import pytest

from agent_eval.runner.check_determinism import check_task_determinism

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"
TASK_NAMES = ["task_01_leakage", "task_02_ambiguous_spec", "task_03_conventions", "task_04_nondeterminism"]


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_task_grader_determinism(task_name: str):
    task_dir = TASKS_DIR / task_name
    is_deterministic = check_task_determinism(
        task_dir=task_dir,
        iterations=5,
        timeout_seconds=40,
    )
    assert is_deterministic is True, f"Grader for {task_name} exhibited non-deterministic behavior over 5 runs."
