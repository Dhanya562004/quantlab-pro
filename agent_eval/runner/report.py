"""
Generates evaluation pass-rate summary reports from runs/*.json.

STRICT RULE:
Only reports real numbers from actual recorded runs in agent_eval/runs/.
Never invents pass rates, transcripts, or failure reasons.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def is_valid_evaluation_run(data: dict[str, Any]) -> bool:
    """
    Checks if a run record is a valid evaluation run.
    Excludes preliminary pilot runs and harness errors.
    Includes valid passed and failed evaluation runs.
    """
    if not isinstance(data, dict):
        return False
    if "task_id" not in data or "agent_name" not in data or "passed" not in data:
        return False

    # Exclude preliminary pilot runs
    run_id = data.get("run_id", "")
    if run_id == "pilot_run_01" or run_id.startswith("pilot_"):
        return False

    # Exclude harness errors (parsing failures, malformed output, etc.)
    status = data.get("status", "")
    if status == "harness_error":
        return False
    error_message = data.get("error_message") or ""
    if error_message.startswith("harness_error:"):
        return False

    return True


def load_all_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """Loads all valid run JSON files from runs_dir."""
    runs = []
    if not runs_dir.exists():
        return runs

    for json_file in sorted(runs_dir.rglob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if is_valid_evaluation_run(data):
                runs.append(data)
        except Exception:
            continue
    return runs


def generate_report(runs_dir: Path | None = None) -> str:
    """Generates a text/markdown pass-rate table aggregated by task and agent."""
    if runs_dir is None:
        runs_dir = Path(__file__).resolve().parent.parent / "runs"

    runs = load_all_runs(runs_dir)

    if not runs:
        return (
            "# Agent Evaluation Run Report\n\n"
            "> [!NOTE]\n"
            "> **Zero runs recorded in `agent_eval/runs/`.**\n"
            "> To generate real data, execute runs with:\n"
            "> `python -m agent_eval.runner.run_task --task task_01_leakage --solution <path> --record`\n\n"
            "| Task | Agent | Runs | Passes | Pass Rate | Main Failure Reason |\n"
            "|---|---|---|---|---|---|\n"
            "| task_01_leakage | TODO | 0 | 0 | TODO: no runs yet | TODO: no runs yet |\n"
            "| task_02_ambiguous_spec | TODO | 0 | 0 | TODO: no runs yet | TODO: no runs yet |\n"
            "| task_03_conventions | TODO | 0 | 0 | TODO: no runs yet | TODO: no runs yet |\n"
            "| task_04_nondeterminism | TODO | 0 | 0 | TODO: no runs yet | TODO: no runs yet |\n"
        )

    # Group runs by (task_id, agent_name)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        grouped[(r["task_id"], r["agent_name"])].append(r)

    lines = [
        "# Agent Evaluation Run Report",
        "",
        f"Total Recorded Runs: {len(runs)}",
        "",
        "| Task | Agent | Runs | Passes | Pass Rate | Main Failure Reason |",
        "|---|---|:---:|:---:|:---:|---|",
    ]

    for (task_id, agent_name), task_runs in sorted(grouped.items()):
        total = len(task_runs)
        passes = sum(1 for r in task_runs if r.get("passed", False))
        rate_str = f"{(passes / total) * 100:.1f}%"

        # Extract common failure reasons
        failures = [r for r in task_runs if not r.get("passed", False)]
        if not failures:
            main_failure = "None (100% pass)"
        else:
            reasons = [f.get("error_message") or "Assertion/Test Failed" for f in failures]
            # Take most frequent
            reason_counts = defaultdict(int)
            for r in reasons:
                reason_counts[r] += 1
            most_common = max(reason_counts.items(), key=lambda x: x[1])[0]
            main_failure = most_common.replace("\n", " ")[:60]

        lines.append(f"| `{task_id}` | `{agent_name}` | {total} | {passes} | {rate_str} | {main_failure} |")

    return "\n".join(lines)


def main() -> None:
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    report_text = generate_report(runs_dir)
    print(report_text)


if __name__ == "__main__":
    main()
