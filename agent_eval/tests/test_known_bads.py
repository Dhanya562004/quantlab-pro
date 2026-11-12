"""
Grader Self-Test: Every known-bad solution must fail grader checks.
"""

from pathlib import Path

import pytest

from agent_eval.runner.agents import DirectoryOverlayRunner
from agent_eval.runner.run_task import run_task

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"

# Gather all known-bad folders across tasks
KNOWN_BAD_CASES = []
for task_dir in sorted(TASKS_DIR.iterdir()):
    if task_dir.is_dir() and task_dir.name.startswith("task_"):
        kb_dir = task_dir / "known_bad"
        if kb_dir.exists():
            for bad_case in sorted(kb_dir.iterdir()):
                if bad_case.is_dir():
                    KNOWN_BAD_CASES.append((task_dir.name, bad_case.name, bad_case))


@pytest.mark.parametrize("task_name,case_name,case_path", KNOWN_BAD_CASES)
def test_known_bad_solution_fails(task_name: str, case_name: str, case_path: Path):
    task_dir = TASKS_DIR / task_name
    runner = DirectoryOverlayRunner(case_path, name=case_name)
    result = run_task(task_dir=task_dir, agent_or_solution=runner, timeout_seconds=45)

    assert result.passed is False, (
        f"Known-bad case '{case_name}' in task '{task_name}' unexpectedly passed grader checks! "
        "The grader must catch this flaw."
    )
