"""
Determinism verification script for task graders.

Runs the grader on the same solution N times (default: 20) and fails if results differ.
Ensures no flaky tests, non-deterministic RNG, race conditions, or unseeded operations.
"""

import argparse
import sys
from pathlib import Path

from agent_eval.runner.agents import DirectoryOverlayRunner
from agent_eval.runner.run_task import run_task


def check_task_determinism(
    task_dir: Path,
    solution_dir: Path | None = None,
    iterations: int = 20,
    timeout_seconds: int = 40,
) -> bool:
    """
    Executes the grader on the specified solution for `iterations` times.
    Returns True if all runs produce identical outcomes, False otherwise.
    """
    task_dir = Path(task_dir).resolve()
    if solution_dir is None:
        solution_dir = task_dir / "reference_solution"

    if not solution_dir.exists():
        raise FileNotFoundError(f"Solution directory not found: {solution_dir}")

    print(f"Checking determinism for {task_dir.name} over {iterations} iterations...")
    runner = DirectoryOverlayRunner(solution_dir, name="determinism_check")

    results = []
    for i in range(iterations):
        res = run_task(
            task_dir=task_dir,
            agent_or_solution=runner,
            run_id=f"det_run_{i+1}",
            timeout_seconds=timeout_seconds,
            record=False,
        )
        summary = {
            "iteration": i + 1,
            "passed": res.passed,
            "tests_passed": res.tests_passed,
            "tests_failed": res.tests_failed,
            "tests_total": res.tests_total,
        }
        results.append(summary)
        print(f"  [{i+1:02d}/{iterations:02d}] passed={res.passed} ({res.tests_passed}/{res.tests_total} passed)")

    # Compare all runs against run 0
    baseline = results[0]
    mismatches = []
    for res in results[1:]:
        if (
            res["passed"] != baseline["passed"]
            or res["tests_passed"] != baseline["tests_passed"]
            or res["tests_failed"] != baseline["tests_failed"]
            or res["tests_total"] != baseline["tests_total"]
        ):
            mismatches.append(res)

    if mismatches:
        print(f"\n[FAIL] Non-determinism detected in {task_dir.name}!")
        print(f"Baseline (Run 1): {baseline}")
        for m in mismatches:
            print(f"Mismatch at Run {m['iteration']}: {m}")
        return False

    print(f"[SUCCESS] {task_dir.name} is 100% deterministic across {iterations} iterations.\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Check determinism of agent evaluation graders.")
    parser.add_argument("--task", help="Specific task name (e.g. task_01_leakage). If omitted, checks all tasks.")
    parser.add_argument("-n", "--iterations", type=int, default=20, help="Number of repetitions (default: 20)")
    parser.add_argument("--timeout", type=int, default=40, help="Grader timeout in seconds per run")

    args = parser.parse_args()

    base_tasks_dir = Path(__file__).resolve().parent.parent / "tasks"
    if args.task:
        task_dirs = [base_tasks_dir / args.task]
    else:
        task_dirs = sorted([d for d in base_tasks_dir.iterdir() if d.is_dir() and d.name.startswith("task_")])

    if not task_dirs:
        sys.exit("No task directories found to evaluate.")

    all_passed = True
    for tdir in task_dirs:
        ok = check_task_determinism(
            task_dir=tdir,
            iterations=args.iterations,
            timeout_seconds=args.timeout,
        )
        if not ok:
            all_passed = False

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
