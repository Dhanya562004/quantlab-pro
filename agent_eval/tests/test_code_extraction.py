"""
Tests for agent code extraction, error handling, and harness_error status reporting.
"""

from pathlib import Path
from typing import Any

import pytest

from agent_eval.runner.agents import (
    AgentRunner,
    HarnessParsingError,
    extract_code_blocks,
)
from agent_eval.runner.run_task import run_task

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


def test_extract_single_fenced_block():
    sample_text = (
        "Here is the solution to the task:\n\n"
        "```python indicators.py\n"
        "import numpy as np\n"
        "def compute_rolling_volatility():\n"
        "    pass\n"
        "```\n"
        "Hope this helps!"
    )
    extracted = extract_code_blocks(sample_text)
    assert "indicators.py" in extracted
    assert "def compute_rolling_volatility" in extracted["indicators.py"]


def test_extract_multiple_files():
    sample_text = (
        "Here are the two required files:\n\n"
        "```python indicators.py\n"
        "def compute_rolling_volatility(): pass\n"
        "```\n\n"
        "```markdown NOTES.md\n"
        "## Technical Assumptions\n"
        "- closed='right'\n"
        "```\n"
    )
    extracted = extract_code_blocks(sample_text)
    assert len(extracted) == 2
    assert "indicators.py" in extracted
    assert "NOTES.md" in extracted
    assert "## Technical Assumptions" in extracted["NOTES.md"]


def test_extract_first_line_comment_path():
    sample_text = (
        "```python\n"
        "# feature_pipeline.py\n"
        "def compute_features(): pass\n"
        "```\n"
    )
    extracted = extract_code_blocks(sample_text)
    assert "feature_pipeline.py" in extracted


def test_extract_no_code_raises_harness_error():
    sample_text = (
        "I have analyzed the technical requirements carefully. You should modify "
        "the function to calculate sample standard deviation with ddof=1 and preserve NaNs."
    )
    with pytest.raises(HarnessParsingError) as exc_info:
        extract_code_blocks(sample_text)
    assert "No fenced code blocks found" in str(exc_info.value)


def test_extract_malformed_fences_raises_harness_error():
    # Odd number of backticks (unclosed fence)
    sample_text = (
        "```python indicators.py\n"
        "def compute_rolling_volatility(): pass\n"
        "# missing closing fence"
    )
    with pytest.raises(HarnessParsingError) as exc_info:
        extract_code_blocks(sample_text)
    assert "odd number of code fence markers" in str(exc_info.value)


class MalformedOutputAgentRunner(AgentRunner):
    """Simulates an agent that outputs pure prose without code fences."""

    def __init__(self):
        super().__init__(name="malformed_agent")

    def solve(self, prompt: str, workspace_dir: Path) -> dict[str, Any]:
        raw_reply = "I cannot solve this because I do not output code fences."
        # Attempt to extract code blocks, triggering HarnessParsingError
        extract_code_blocks(raw_reply)
        return {"agent": self.name}


def test_harness_error_recorded_in_run_result():
    """
    Verify that parsing failures are classified as status='harness_error',
    NOT as regular test failures.
    """
    task_dir = TASKS_DIR / "task_01_leakage"
    runner = MalformedOutputAgentRunner()

    result = run_task(task_dir=task_dir, agent_or_solution=runner, timeout_seconds=10)

    assert result.passed is False
    assert result.status == "harness_error"
    assert "harness_error" in result.error_message
    assert result.tests_failed == 0, "Harness parsing errors must not count as failed grader tests"
