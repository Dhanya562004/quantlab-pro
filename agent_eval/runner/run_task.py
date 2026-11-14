"""
Task execution runner for agent evaluation.

Copies starter to an isolated temp directory, applies an agent's solution
(via AgentRunner, directory overlay, or import), executes hidden pytest grader with
timeouts, evaluates rubric conditions, and writes structured execution JSON.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_eval.runner.agents import (
    AgentRunner,
    DirectoryOverlayRunner,
    HarnessParsingError,
    ImportAgentRunner,
    LLMAgentRunner,
    list_provider_models,
)


@dataclass
class EvaluationResult:
    task_id: str
    agent_name: str
    run_id: str
    timestamp: str
    passed: bool
    tests_passed: int
    tests_failed: int
    tests_total: int
    duration_seconds: float
    error_message: str | None = None
    test_output: str = ""
    rubric_results: dict[str, Any] = field(default_factory=dict)
    files_modified: list[str] = field(default_factory=list)
    status: str = "completed"  # "passed", "failed", "harness_error"
    model_name: str | None = None
    mode: str = "oneshot"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_task(
    task_dir: Path,
    agent_or_solution: AgentRunner | Path,
    run_id: str | None = None,
    timeout_seconds: int = 45,
    record: bool = False,
    runs_dir: Path | None = None,
) -> EvaluationResult:
    """
    Executes a task against a given solution/runner in an isolated temporary directory.
    """
    task_dir = Path(task_dir).resolve()
    task_id = task_dir.name
    starter_dir = task_dir / "starter"
    grader_dir = task_dir / "grader"
    prompt_file = task_dir / "prompt.md"

    if not starter_dir.exists():
        raise FileNotFoundError(f"Starter directory not found at: {starter_dir}")
    if not grader_dir.exists():
        raise FileNotFoundError(f"Grader directory not found at: {grader_dir}")
    if not prompt_file.exists():
        raise FileNotFoundError(f"prompt.md not found at: {prompt_file}")

    prompt_text = prompt_file.read_text(encoding="utf-8")
    run_id_val = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # Determine AgentRunner instance
    if isinstance(agent_or_solution, AgentRunner):
        runner = agent_or_solution
    elif isinstance(agent_or_solution, (str, Path)):
        sol_path = Path(agent_or_solution).resolve()
        runner = DirectoryOverlayRunner(sol_path, name=sol_path.name)
    else:
        raise TypeError(f"Expected AgentRunner or Path, got {type(agent_or_solution)}")

    agent_name = runner.name
    start_time = time.perf_counter()
    transcript = ""

    with tempfile.TemporaryDirectory(prefix=f"eval_{task_id}_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # 1. Copy starter repo into temp workspace
        for item in starter_dir.iterdir():
            if item.name == "__pycache__":
                continue
            if item.is_dir():
                shutil.copytree(item, temp_dir / item.name)
            else:
                shutil.copy2(item, temp_dir / item.name)

        # 2. Apply Agent Solution
        try:
            solve_meta = runner.solve(prompt=prompt_text, workspace_dir=temp_dir)
            files_modified = solve_meta.get("files_modified", [])
            transcript = solve_meta.get("transcript", "")
            exact_model = solve_meta.get("model")
            exec_mode = solve_meta.get("mode", getattr(runner, "mode", "oneshot"))
        except HarnessParsingError as hpe:
            duration = time.perf_counter() - start_time
            res = EvaluationResult(
                task_id=task_id,
                agent_name=agent_name,
                run_id=run_id_val,
                timestamp=datetime.now(timezone.utc).isoformat(),
                passed=False,
                tests_passed=0,
                tests_failed=0,
                tests_total=0,
                duration_seconds=duration,
                error_message=f"harness_error: {hpe}",
                test_output=f"Harness parsing failure:\n{hpe}",
                rubric_results={"solution_applied": False},
                files_modified=[],
                status="harness_error",
                model_name=getattr(runner, "model_name", None),
                mode=getattr(runner, "mode", "oneshot"),
            )
            if record:
                _save_run_result(res, transcript, runs_dir)
            return res
        except Exception as exc:
            duration = time.perf_counter() - start_time
            res = EvaluationResult(
                task_id=task_id,
                agent_name=agent_name,
                run_id=run_id_val,
                timestamp=datetime.now(timezone.utc).isoformat(),
                passed=False,
                tests_passed=0,
                tests_failed=1,
                tests_total=1,
                duration_seconds=duration,
                error_message=f"Agent failed to apply solution: {exc}",
                test_output=f"Exception during solve step:\n{exc}",
                rubric_results={"solution_applied": False},
                files_modified=[],
                status="failed",
                model_name=getattr(runner, "model_name", None),
                mode=getattr(runner, "mode", "oneshot"),
            )
            if record:
                _save_run_result(res, transcript, runs_dir)
            return res

        # 3. Copy grader into temp directory to execute
        temp_grader_dir = temp_dir / "_eval_grader"
        shutil.copytree(grader_dir, temp_grader_dir)

        # 4. Run pytest against the grader tests
        test_file = temp_grader_dir / "test_grader.py"
        if not test_file.exists():
            test_candidates = list(temp_grader_dir.glob("test_*.py"))
            if not test_candidates:
                raise FileNotFoundError(f"No test_*.py found in grader dir {grader_dir}")
            test_file = test_candidates[0]

        env = os.environ.copy()
        # Prepend workspace to PYTHONPATH
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{temp_dir}{os.pathsep}{existing_pythonpath}"

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            "-v",
            "--tb=short",
            "-o",
            "pythonpath=.",
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(temp_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            raw_stdout = proc.stdout + "\n" + proc.stderr
            pytest_passed = (proc.returncode == 0)
        except subprocess.TimeoutExpired as te:
            pytest_passed = False
            raw_stdout = f"Execution timed out after {timeout_seconds}s.\nStdout:\n{te.stdout}\nStderr:\n{te.stderr}"
        except Exception as exc:
            pytest_passed = False
            raw_stdout = f"Grader failed to execute: {exc}"

        duration = time.perf_counter() - start_time

        # Parse test metrics from stdout
        passed_count = raw_stdout.count(" PASSED")
        failed_count = raw_stdout.count(" FAILED")
        error_count = raw_stdout.count(" ERROR")
        total_tests = passed_count + failed_count + error_count
        if total_tests == 0:
            total_tests = 1 if not pytest_passed else 0

        # Evaluate rubric requirements if present
        rubric_results = _evaluate_rubric(temp_dir, grader_dir)
        rubric_passed = rubric_results.get("all_passed", True)

        overall_passed = bool(pytest_passed and rubric_passed)

        result = EvaluationResult(
            task_id=task_id,
            agent_name=agent_name,
            run_id=run_id_val,
            timestamp=datetime.now(timezone.utc).isoformat(),
            passed=overall_passed,
            tests_passed=passed_count,
            tests_failed=failed_count + error_count,
            tests_total=total_tests,
            duration_seconds=duration,
            error_message=None if overall_passed else "One or more grader tests/rubrics failed.",
            test_output=raw_stdout,
            rubric_results=rubric_results,
            files_modified=files_modified,
            status="passed" if overall_passed else "failed",
            model_name=exact_model or getattr(runner, "model_name", None),
            mode=exec_mode,
        )

        if record:
            _save_run_result(result, transcript, runs_dir)

        return result


def _evaluate_rubric(workspace_dir: Path, grader_dir: Path) -> dict[str, Any]:
    """Evaluates procedural rubric rules defined for the task."""
    rubric_file = grader_dir / "rubric.md"
    if not rubric_file.exists():
        return {"all_passed": True}

    results: dict[str, Any] = {"all_passed": True, "checks": {}}
    content = rubric_file.read_text(encoding="utf-8")

    # If rubric specifies that NOTES.md must be documented:
    if "NOTES.md" in content:
        notes_path = workspace_dir / "NOTES.md"
        exists = notes_path.exists()
        has_content = False
        if exists:
            text = notes_path.read_text(encoding="utf-8").strip()
            has_content = len(text) > 20
        check_ok = exists and has_content
        results["checks"]["notes_documented"] = check_ok
        if not check_ok:
            results["all_passed"] = False

    return results


def _save_run_result(res: EvaluationResult, transcript: str, runs_dir: Path | None) -> Path:
    """Saves run JSON record and diff/transcript to runs/<task>/<agent>/<run_id>.json."""
    if runs_dir is None:
        runs_dir = Path(__file__).resolve().parent.parent / "runs"
    task_agent_dir = runs_dir / res.task_id / res.agent_name
    task_agent_dir.mkdir(parents=True, exist_ok=True)

    json_file = task_agent_dir / f"{res.run_id}.json"
    json_file.write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")

    if transcript:
        transcript_file = task_agent_dir / f"{res.run_id}_transcript.txt"
        transcript_file.write_text(transcript, encoding="utf-8")

    return json_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an evaluation task against an agent solution.")
    parser.add_argument("--task", help="Task name or path (e.g. task_01_leakage)")
    parser.add_argument("--solution", help="Path to solution directory or diff file")
    parser.add_argument("--agent", default="import", choices=["import", "llm", "overlay"])
    parser.add_argument("--mode", default="oneshot", choices=["oneshot", "agent"], help="Agent interaction mode")
    parser.add_argument("--model", help="LLM model name if using --agent llm")
    parser.add_argument("--run-id", help="Optional run id string")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--max-steps", type=int, default=15, help="Maximum steps in agent loop")
    parser.add_argument("--runs", type=int, default=1, help="Number of evaluation runs")
    parser.add_argument("--record", action="store_true", help="Record result into runs/ directory")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and token/cost estimate without calling API")
    parser.add_argument(
        "--list-models",
        nargs="?",
        const="all",
        help="List available model names from provider API (openai, gemini, anthropic)",
    )

    args = parser.parse_args()

    # Handle --list-models flag
    if args.list_models:
        prov = None if args.list_models == "all" else args.list_models
        models_by_provider = list_provider_models(provider=prov)
        print("=== Real Provider Models Available ===")
        for p_name, model_list in models_by_provider.items():
            print(f"\n[{p_name.upper()}] ({len(model_list)} models):")
            for m in model_list[:15]:
                print(f"  - {m}")
            if len(model_list) > 15:
                print(f"  ... and {len(model_list) - 15} more.")
        return

    if not args.task:
        sys.exit("Error: --task is required (or specify --list-models).")

    # Resolve task dir
    base_tasks_dir = Path(__file__).resolve().parent.parent / "tasks"
    task_dir = Path(args.task)
    if not task_dir.is_absolute():
        if (base_tasks_dir / args.task).exists():
            task_dir = base_tasks_dir / args.task
        elif not task_dir.exists():
            sys.exit(f"Task directory not found: {args.task}")

    # Handle --dry-run flag
    if args.dry_run:
        prompt_chars = (
            len(task_dir.joinpath("prompt.md").read_text(encoding="utf-8"))
            if task_dir.joinpath("prompt.md").exists()
            else 0
        )
        starter_chars = sum(
            len(f.read_text(encoding="utf-8", errors="ignore"))
            for f in task_dir.joinpath("starter").rglob("*")
            if f.is_file()
        )
        est_tokens = (prompt_chars + starter_chars) // 4
        model_display = args.model or "default provider model"
        print("============================= Dry Run Plan =============================")
        print(f"Task:                    {task_dir.name}")
        print(f"Task Directory:          {task_dir}")
        print(f"Execution Mode:          {args.mode}")
        print(f"Agent Architecture:      {args.agent}")
        print(f"Model:                   {model_display}")
        print(f"Number of Planned Runs:  {args.runs}")
        print(f"Max Steps per Run:       {args.max_steps if args.mode == 'agent' else 1}")
        print(f"Timeout per Run:         {args.timeout}s")
        print(f"Est. Context Tokens:     ~{est_tokens:,} tokens per run")
        print("Est. Cost per Run:       ~$0.01 - $0.05")
        print(f"Total Est. Cost:         ~${0.01 * args.runs:.2f} - ${0.05 * args.runs:.2f}")
        print("Status:                  DRY RUN (no network calls made, no credits consumed)")
        print("========================================================================")
        return

    if args.agent == "llm":
        runner: AgentRunner = LLMAgentRunner(
            model_name=args.model,
            mode=args.mode,
            max_steps=args.max_steps,
            timeout_seconds=args.timeout,
        )
    elif args.solution:
        sol_path = Path(args.solution).resolve()
        if args.agent == "import":
            runner = ImportAgentRunner(sol_path)
        else:
            runner = DirectoryOverlayRunner(sol_path)
    else:
        # Default to reference solution if available
        ref_sol = task_dir / "reference_solution"
        if ref_sol.exists():
            runner = DirectoryOverlayRunner(ref_sol, name="reference_solution")
        else:
            sys.exit("Error: Must provide --solution or use --agent llm")

    res = run_task(
        task_dir=task_dir,
        agent_or_solution=runner,
        run_id=args.run_id,
        timeout_seconds=args.timeout,
        record=args.record,
    )

    print(f"\nTask: {res.task_id} | Agent: {res.agent_name} | Run: {res.run_id}")
    print(f"Status: {res.status.upper()} (Duration: {res.duration_seconds:.2f}s)")
    print(f"Tests: {res.tests_passed} passed, {res.tests_failed} failed, {res.tests_total} total")
    if not res.passed:
        print("\n--- Grader Output ---")
        print(res.test_output)
        sys.exit(1)


if __name__ == "__main__":
    main()
