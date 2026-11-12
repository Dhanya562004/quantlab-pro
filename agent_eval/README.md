# 🧪 Agent Evaluation Harness

A test-driven evaluation harness for measuring coding agent performance on hard, realistic quantitative software engineering tasks. Built on the RL-task builder paradigm: author hard tasks, grade them procedurally with hidden assertions, run agents repeatedly across independent attempts, and analyze failure patterns without inventing results.

---

## 📁 Architecture & Task Structure

```text
agent_eval/
  README.md                  # System overview, execution instructions, grading rules, limits
  FAILURE_ANALYSIS.md        # Empirical failure analysis filled strictly from real runs
  tasks/
    task_01_leakage/         # Look-ahead data leakage in feature pipeline
    task_02_ambiguous_spec/  # Ambiguous boundary conventions & explicit assumptions
    task_03_conventions/     # Cross-cutting repo conventions (adding a 7th agent)
    task_04_nondeterminism/  # Non-deterministic training & multi-RNG synchronization
      Each task contains:
        prompt.md            # Exact task prompt (contains every fact needed to pass)
        starter/             # Baseline repository code the agent edits
        grader/              # Hidden pytest test suite + rubric.md
        reference_solution/  # Verified correct solution used to test the grader
        known_bad/           # 2-3 plausible incorrect solutions used to test the grader
        TASK_NOTES.md        # Technical breakdown of the separate sources of difficulty
  runner/
    run_task.py              # Copies starter to temp dir, applies solution, runs grader under timeout
    agents.py                # AgentRunner interfaces (LLM API runner and import runner)
    check_determinism.py     # Executes grader on the same solution N times (fails if results differ)
    report.py                # Reads runs/*.json and prints empirical pass-rate table
  runs/                      # Stored execution records, transcripts, and diffs (.json)
  tests/                     # CI test suite for the evaluation harness itself
```

---

## 🎯 The Four Evaluation Tasks

Each task is constructed with **at least two distinct sources of difficulty** designed to probe real agent failure modes:

| Task | Domain | Sources of Difficulty | Verification Method |
|---|---|---|---|
| **`task_01_leakage`** | Feature Engineering | 1. Centered rolling window (`center=True`) looks into future prices.<br>2. Backward-fill (`bfill`) propagates future valid values into warmup periods.<br>3. Public API signatures and column names must remain unchanged. | **Shifted-Target / Future Perturbation Test:** Mutating future prices must produce bit-for-bit identical past features. |
| **`task_02_ambiguous_spec`** | Financial Indicators | 1. Bessel's correction (`ddof=1`), flat price zero-variance handling, and negative price rejection.<br>2. Window closure (`closed='right'`), warmup NaN preservation, and annualization ($\sqrt{252}$) must be documented in `NOTES.md`. | **Procedural Tests + Rubric:** Mathematical edge cases evaluated in pytest; rubric asserts `NOTES.md` exists and documents assumptions. |
| **`task_03_conventions`** | Pipeline Architecture | 1. Adding a 7th agent requires synchronized edits across 6 files (`state.py`, `stress_agent.py`, `agents/__init__.py`, `registry.py`, `engine.py`, `docs/agents.md`).<br>2. State mutation must use typed Pydantic `StressReport` model with null-safety. | **Convention Grader:** Validates Pydantic schema, allowlisted registry lookup, orchestrator execution trace, unit tests, and docs. |
| **`task_04_nondeterminism`** | ML Training | 1. Unordered `set()` iteration randomizes neural network weight assignment across Python runs.<br>2. PyTorch DataLoader shuffle without dedicated seeded generator drifts across batches.<br>3. Multi-library seeding (`random`, `numpy`, `torch`). | **Hash Determinism:** Running training twice with `seed=42` must yield bit-for-bit identical parameter SHA-256 hashes. |

---

## 🚀 How to Run

### 1. Evaluate a Solution Against a Task
Execute a task evaluation in a clean, isolated temporary workspace:
```bash
# Evaluate using reference solution
python -m agent_eval.runner.run_task --task task_01_leakage --solution agent_eval/tasks/task_01_leakage/reference_solution

# Evaluate a solution and record results to runs/
python -m agent_eval.runner.run_task --task task_01_leakage --solution agent_eval/tasks/task_01_leakage/reference_solution --record --run-id run_01
```

### 2. Import External Agent Runs
To evaluate solutions produced by hand or generated with an external agent:
```bash
python -m agent_eval.runner.run_task --task task_01_leakage --agent import --solution path/to/saved_solution --record
```

### 3. Run Live LLM Coding Agents
Using environment keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`):
```bash
export GEMINI_API_KEY="your-key-here"
python -m agent_eval.runner.run_task --task task_01_leakage --agent llm --model gemini-2.5-pro --record
```

### 4. Check Grader Determinism (N Iterations)
Run the grader N times (default 20, or N=5 for fast check) to guarantee zero test flakiness:
```bash
# Check all tasks over 20 iterations
python -m agent_eval.runner.check_determinism -n 20

# Check specific task
python -m agent_eval.runner.check_determinism --task task_01_leakage -n 20
```

### 5. Generate Empirical Pass-Rate Report
Reads all run records from `agent_eval/runs/` and prints a structured summary:
```bash
python -m agent_eval.runner.report
```

---

## 📏 Grader Rules & Integrity Guarantees

1. **Procedural PyTest First**: All functional correctness, numerical tolerances, and structural contracts are tested via deterministic procedural assertions in `grader/test_grader.py`.
2. **Deterministic & Isolated**:
   - Every run executes in a fresh `tempfile.TemporaryDirectory`.
   - Fixed random seeds; zero external network access.
   - Per-test subprocess execution timeouts prevent infinite loops.
3. **Fairness Guarantee**:
   - Grader self-tests include a **fairness test** (`test_fairness.py`) for every task.
   - It asserts that `prompt.md` explicitly specifies every public function signature, parameter name, column name, and convention evaluated by the hidden tests. No secret requirements or trick traps.
4. **Grader Triad Verification**:
   - Reference solution must pass 100% of tests.
   - Every known-bad solution must fail the grader.
   - Determinism checks verify that results never fluctuate across repeated executions.
5. **No Fabricated Numbers**:
   - Pass rates and transcripts are populated strictly from real runs saved in `runs/`. Unrun tasks are explicitly marked `TODO`.

---

## ⚠️ Known Limits

- **Probe-Scale Tasks, Not a General Benchmark**: These four tasks evaluate specific high-frequency failure modes in quantitative programming (leakage, boundary ambiguity, conventions, determinism). They are diagnostic probes, not a broad foundation model benchmark.
- **Rubric Scope**: Procedural tests handle all executable assertions. Free-form text rubrics in `rubric.md` are reserved for non-executable documentation checks and are not part of automated CI.

