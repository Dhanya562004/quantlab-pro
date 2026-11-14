"""
Security and Isolation Test:
Proves that the hidden grader never exists in the agent workspace during execution.
"""

from pathlib import Path
from typing import Any

from agent_eval.runner.agents import AgentRunner
from agent_eval.runner.run_task import run_task

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


class WorkspaceSpyRunner(AgentRunner):
    """
    An agent that inspects its entire workspace during solve() to check for grader leakage.
    """

    def __init__(self):
        super().__init__(name="workspace_spy")
        self.observed_files: list[str] = []
        self.grader_detected: bool = False

    def solve(self, prompt: str, workspace_dir: Path) -> dict[str, Any]:
        # Recursively list all files and directories in workspace
        all_items = list(workspace_dir.rglob("*"))
        item_names = [p.name for p in all_items]
        self.observed_files = [str(p.relative_to(workspace_dir)) for p in all_items if p.is_file()]

        # Check for hidden grader files or directories
        if "test_grader.py" in item_names or "rubric.md" in item_names or "_eval_grader" in item_names:
            self.grader_detected = True

        # Check for any file containing "grader"
        for name in item_names:
            if "grader" in name.lower():
                self.grader_detected = True

        return {
            "agent": self.name,
            "type": "spy",
            "files_modified": [],
            "transcript": "Inspected workspace for hidden grader files.",
        }


def test_hidden_grader_not_in_workspace_during_solve():
    """
    Verify that when an agent's solve() method executes, no grader files exist
    in its workspace. The hidden grader is only mounted after agent completion.
    """
    task_dir = TASKS_DIR / "task_01_leakage"
    spy = WorkspaceSpyRunner()

    # Execute task
    run_task(task_dir=task_dir, agent_or_solution=spy, timeout_seconds=10)

    # Prove grader was completely invisible to the agent during solve
    assert not spy.grader_detected, (
        f"Security violation! Hidden grader files were detected in workspace: {spy.observed_files}"
    )

    # Prove only starter files were present in workspace
    starter_dir = task_dir / "starter"
    expected_starter_files = {
        str(p.relative_to(starter_dir)).replace("\\", "/")
        for p in starter_dir.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    observed_files_norm = {p.replace("\\", "/") for p in spy.observed_files}

    assert observed_files_norm == expected_starter_files, (
        f"Workspace files {observed_files_norm} did not match starter {expected_starter_files}"
    )

