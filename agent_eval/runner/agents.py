"""
Agent runner interfaces and implementations for agent evaluation.

Supports:
- ImportAgentRunner: loads hand-crafted or pre-generated solutions from disk/diffs.
- DirectoryOverlayRunner: applies a local folder overlay (used for reference/known-bad testing).
- LLMAgentRunner: calls an LLM provider (OpenAI, Anthropic, or Gemini) using environment keys.
  Supports both oneshot mode and interactive agent mode with tool use.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class HarnessParsingError(Exception):
    """Raised when agent solution output cannot be parsed into valid code/files."""


def extract_code_blocks(text: str) -> dict[str, str]:
    """
    Extracts code blocks with explicit target file paths from markdown text.

    Supports formats:
    - ```python filepath.py\n...```
    - ```python\n# filepath.py\n...```
    - ```markdown NOTES.md\n...```
    - ```filepath.py\n...```

    Raises HarnessParsingError if:
    - No code blocks exist in the text.
    - Code fences are malformed or unclosed.
    - Fenced blocks contain no valid target file path.
    """
    if not text or not text.strip():
        raise HarnessParsingError("Empty agent response: no code blocks found.")

    # Check for unclosed fence: odd number of triple backticks
    fence_count = text.count("```")
    if fence_count % 2 != 0:
        raise HarnessParsingError(
            f"Malformed markdown: found odd number of code fence markers ({fence_count})."
        )

    if fence_count == 0:
        raise HarnessParsingError("No fenced code blocks found in agent response.")

    pattern = re.compile(r"```([^\r\n]*)\r?\n(.*?)```", re.DOTALL)
    matches = list(pattern.finditer(text))
    if not matches:
        raise HarnessParsingError("No valid fenced code blocks found in agent response.")

    extracted_files: dict[str, str] = {}

    for match in matches:
        header = match.group(1).strip()
        body = match.group(2)

        filepath: str | None = None

        # Case 1: Header has path after language: ```python path/to/file.py
        header_tokens = header.split()
        if len(header_tokens) > 1:
            # Second token or token with extension/slash
            for token in header_tokens[1:]:
                clean_token = token.strip("`'\" :")
                if "." in clean_token or "/" in clean_token or "\\" in clean_token:
                    filepath = clean_token
                    break
            if not filepath:
                filepath = header_tokens[1].strip("`'\" :")
        elif len(header_tokens) == 1:
            token = header_tokens[0].strip("`'\" :")
            # If the single token looks like a file name (e.g. indicators.py or NOTES.md)
            if "." in token and token.lower() not in {"py", "md", "sh", "txt"}:
                filepath = token

        # Case 2: Filepath is specified on the first line of the code block: # filepath.py
        if not filepath:
            lines = body.strip().splitlines()
            if lines:
                first_line = lines[0].strip()
                match_comment = re.match(r"^(?:#|//|<!--)\s*([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9_]+)", first_line)
                if match_comment:
                    filepath = match_comment.group(1)

        if filepath:
            # Normalize path
            norm_path = filepath.replace("\\", "/").strip("/")
            extracted_files[norm_path] = body

    if not extracted_files:
        raise HarnessParsingError(
            "Found code blocks, but could not determine target file path for any block. "
            "Ensure code fences specify the relative path on the fence line or first comment line."
        )

    return extracted_files


def list_provider_models(provider: str | None = None) -> dict[str, list[str]]:
    """
    Queries model providers for their actual available model IDs using environment keys.
    Never returns hardcoded or guessed lists.
    """
    results: dict[str, list[str]] = {}

    # 1. OpenAI
    if provider in (None, "openai"):
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                req = urllib.request.Request(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {openai_key}"},
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = sorted([m["id"] for m in data.get("data", [])])
                    results["openai"] = models
            except Exception as exc:
                results["openai"] = [f"Error querying OpenAI API: {exc}"]
        else:
            if provider == "openai":
                results["openai"] = ["Error: OPENAI_API_KEY environment variable not set."]

    # 2. Gemini
    if provider in (None, "gemini"):
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key}"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = sorted([
                        m["name"].replace("models/", "")
                        for m in data.get("models", [])
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                    ])
                    results["gemini"] = models
            except Exception as exc:
                results["gemini"] = [f"Error querying Gemini API: {exc}"]
        else:
            if provider == "gemini":
                results["gemini"] = ["Error: GEMINI_API_KEY environment variable not set."]

    # 3. Anthropic
    if provider in (None, "anthropic"):
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/models",
                    headers={
                        "x-api-key": anthropic_key,
                        "anthropic-version": "2023-06-01",
                    },
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = sorted([m["id"] for m in data.get("data", [])])
                    results["anthropic"] = models
            except Exception as exc:
                results["anthropic"] = [f"Error querying Anthropic API: {exc}"]
        else:
            if provider == "anthropic":
                results["anthropic"] = ["Error: ANTHROPIC_API_KEY environment variable not set."]

    return results


class AgentWorkspaceTools:
    """
    Sandboxed tools exposed to the agent in interactive mode.
    Strictly restricted to the workspace directory.
    """

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir.resolve()

    def _resolve(self, rel_path: str) -> Path:
        resolved = (self.workspace_dir / rel_path).resolve()
        if not str(resolved).startswith(str(self.workspace_dir)):
            raise PermissionError(f"Access denied: path '{rel_path}' is outside the workspace.")
        return resolved

    def list_files(self, directory: str = ".") -> list[str]:
        target = self._resolve(directory)
        if not target.exists() or not target.is_dir():
            return []
        files = []
        for p in sorted(target.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                files.append(str(p.relative_to(self.workspace_dir)).replace("\\", "/"))
        return files

    def read_file(self, filepath: str) -> str:
        target = self._resolve(filepath)
        if not target.exists() or not target.is_file():
            return f"Error: File not found: {filepath}"
        return target.read_text(encoding="utf-8")

    def write_file(self, filepath: str, content: str) -> str:
        target = self._resolve(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {filepath}"

    def run_tests(self, args: str = "") -> str:
        """Runs pytest strictly on visible tests located inside the workspace."""
        # Find test files in workspace
        test_files = list(self.workspace_dir.glob("test_*.py")) + list(self.workspace_dir.glob("tests/**/test_*.py"))
        if not test_files:
            return "No visible test files found in workspace."

        cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"]
        if args:
            cmd.extend(args.split())

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.workspace_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"

        try:
            res = subprocess.run(
                cmd,
                cwd=str(self.workspace_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return (res.stdout + "\n" + res.stderr).strip()
        except Exception as exc:
            return f"Test run error: {exc}"


class AgentRunner(ABC):
    """Abstract interface for agent runners."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def solve(self, prompt: str, workspace_dir: Path) -> dict[str, Any]:
        """
        Apply agent modifications to the workspace_dir according to prompt.
        Returns a dict containing execution metadata and transcripts.
        """


class DirectoryOverlayRunner(AgentRunner):
    """Applies a directory overlay to workspace (used for reference and known-bad tests)."""

    def __init__(self, overlay_dir: Path, name: str = "directory_overlay"):
        super().__init__(name=name)
        self.overlay_dir = Path(overlay_dir)

    def solve(self, prompt: str, workspace_dir: Path) -> dict[str, Any]:
        if not self.overlay_dir.exists():
            raise FileNotFoundError(f"Overlay directory does not exist: {self.overlay_dir}")

        copied_files = []
        for src_path in self.overlay_dir.rglob("*"):
            if src_path.is_file():
                rel_path = src_path.relative_to(self.overlay_dir)
                dest_path = workspace_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest_path)
                copied_files.append(str(rel_path).replace("\\", "/"))

        return {
            "agent": self.name,
            "type": "directory_overlay",
            "source": str(self.overlay_dir),
            "files_modified": copied_files,
            "transcript": f"Applied overlay from {self.overlay_dir}",
        }


class ImportAgentRunner(AgentRunner):
    """
    Imports solutions produced by hand or with another coding agent and saved under runs/.
    Can accept a directory containing files or a patch/diff file.
    """

    def __init__(self, solution_source: Path, name: str = "imported_agent"):
        super().__init__(name=name)
        self.solution_source = Path(solution_source)

    def solve(self, prompt: str, workspace_dir: Path) -> dict[str, Any]:
        if not self.solution_source.exists():
            raise FileNotFoundError(f"Solution source not found: {self.solution_source}")

        if self.solution_source.is_dir():
            overlay = DirectoryOverlayRunner(self.solution_source, name=self.name)
            res = overlay.solve(prompt, workspace_dir)
            res["type"] = "import_directory"
            return res

        if self.solution_source.suffix in {".patch", ".diff"}:
            patch_content = self.solution_source.read_text(encoding="utf-8")
            self._apply_simple_patch(patch_content, workspace_dir)
            return {
                "agent": self.name,
                "type": "import_patch",
                "source": str(self.solution_source),
                "transcript": patch_content,
            }

        if self.solution_source.suffix == ".json":
            data = json.loads(self.solution_source.read_text(encoding="utf-8"))
            if "files" in data and isinstance(data["files"], dict):
                for rel_path_str, content in data["files"].items():
                    target = workspace_dir / rel_path_str
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
                return {
                    "agent": self.name,
                    "type": "import_json_manifest",
                    "source": str(self.solution_source),
                    "files_modified": list(data["files"].keys()),
                    "transcript": data.get("transcript", "Loaded from JSON manifest"),
                }

        raise ValueError(f"Unsupported solution source format: {self.solution_source}")

    def _apply_simple_patch(self, patch_text: str, workspace_dir: Path) -> None:
        file_matches = re.finditer(r"\+\+\+ b/(.*?)\n(.*)", patch_text, re.DOTALL)
        for m in file_matches:
            rel_path = m.group(1).strip()
            dest = workspace_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)


class LLMAgentRunner(AgentRunner):
    """
    Executes an LLM call using credentials from environment variables.
    Supports:
    - mode="oneshot": Single prompt asking model to generate solution files in code fences.
    - mode="agent": Interactive agent loop with list_files, read_file, write_file, run_tests tools.
    """

    def __init__(
        self,
        model_name: str | None = None,
        name: str = "llm_agent",
        mode: str = "oneshot",
        max_steps: int = 15,
        timeout_seconds: int = 300,
    ):
        super().__init__(name=name)
        self.model_name = model_name or os.environ.get("AGENT_EVAL_MODEL", "gpt-4o")
        self.mode = mode
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds

    def solve(self, prompt: str, workspace_dir: Path) -> dict[str, Any]:
        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

        if not (gemini_key or openai_key or anthropic_key):
            raise ValueError(
                "No LLM API key detected in environment. Please set GEMINI_API_KEY, "
                "OPENAI_API_KEY, or ANTHROPIC_API_KEY to run live LLMAgentRunner."
            )

        if self.mode == "agent":
            return self._solve_agent_loop(prompt, workspace_dir)
        return self._solve_oneshot(prompt, workspace_dir)

    def _solve_oneshot(self, prompt: str, workspace_dir: Path) -> dict[str, Any]:
        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

        context_files: dict[str, str] = {}
        for p in workspace_dir.rglob("*.py"):
            if "__pycache__" not in p.parts:
                context_files[str(p.relative_to(workspace_dir)).replace("\\", "/")] = p.read_text(encoding="utf-8")

        for p in workspace_dir.glob("*.md"):
            context_files[str(p.relative_to(workspace_dir)).replace("\\", "/")] = p.read_text(encoding="utf-8")

        prompt_payload = (
            f"You are an expert quantitative software engineering agent.\n"
            f"Here is your assignment:\n\n{prompt}\n\n"
            f"Existing repository files:\n"
        )
        for rel_name, content in context_files.items():
            prompt_payload += f"\n--- {rel_name} ---\n{content}\n"

        prompt_payload += (
            "\nOutput all updated or created files in standard markdown code fences, "
            "with the relative filepath on the fence header line or as a comment on the first line.\n"
            "Example:\n```python filepath/to/file.py\n# code\n```\n"
        )

        response_text = ""
        provider_used = ""
        exact_model = self.model_name

        if openai_key:
            provider_used = "openai"
            response_text, exact_model = self._call_openai(openai_key, prompt_payload)
        elif gemini_key:
            provider_used = "gemini"
            response_text, exact_model = self._call_gemini(gemini_key, prompt_payload)
        elif anthropic_key:
            provider_used = "anthropic"
            response_text, exact_model = self._call_anthropic(anthropic_key, prompt_payload)

        # Parse generated files; raises HarnessParsingError on missing/malformed blocks
        file_map = extract_code_blocks(response_text)
        modified_files = []
        for rel_path_str, code_content in file_map.items():
            target = workspace_dir / rel_path_str
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(code_content, encoding="utf-8")
            modified_files.append(rel_path_str)

        return {
            "agent": self.name,
            "provider": provider_used,
            "model": exact_model,
            "mode": "oneshot",
            "files_modified": modified_files,
            "transcript": response_text,
        }

    def _solve_agent_loop(self, prompt: str, workspace_dir: Path) -> dict[str, Any]:
        """Runs interactive agent loop with tool use."""
        tools = AgentWorkspaceTools(workspace_dir)
        transcript_lines: list[str] = [
            "=== Interactive Agent Session Started ===",
            f"Mode: agent | Max steps: {self.max_steps} | Timeout: {self.timeout_seconds}s",
            f"Task Prompt:\n{prompt}\n",
        ]

        system_instructions = (
            "You are an autonomous coding agent operating in a sandboxed repository workspace.\n"
            "Available tools:\n"
            "1. list_files(directory: str = '.') -> list[str]\n"
            "2. read_file(filepath: str) -> str\n"
            "3. write_file(filepath: str, content: str) -> str\n"
            "4. run_tests(args: str = '') -> str\n"
            "5. finish(message: str) -> str\n\n"
            "To call a tool, respond with ONLY a single JSON object in this format:\n"
            "{\"tool\": \"<tool_name>\", \"args\": {<arguments>}}\n"
            "When done, call finish."
        )

        history: list[dict[str, str]] = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": f"Task instructions:\n{prompt}"},
        ]

        modified_files: set[str] = set()
        start_time = time.perf_counter()
        exact_model = self.model_name
        provider_used = "unknown"

        openai_key = os.environ.get("OPENAI_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

        for step in range(1, self.max_steps + 1):
            if (time.perf_counter() - start_time) > self.timeout_seconds:
                transcript_lines.append(f"Step {step}: Time limit reached ({self.timeout_seconds}s).")
                break

            # Format conversation for LLM call
            combined_prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)

            if openai_key:
                provider_used = "openai"
                resp_text, exact_model = self._call_openai(openai_key, combined_prompt)
            elif gemini_key:
                provider_used = "gemini"
                resp_text, exact_model = self._call_gemini(gemini_key, combined_prompt)
            elif anthropic_key:
                provider_used = "anthropic"
                resp_text, exact_model = self._call_anthropic(anthropic_key, combined_prompt)
            else:
                raise ValueError("No API key available")

            transcript_lines.append(f"\n--- Step {step} [Agent] ---\n{resp_text}")
            history.append({"role": "assistant", "content": resp_text})

            # Parse tool call from response
            tool_name = None
            tool_args = {}
            try:
                # Find JSON block in reply
                json_match = re.search(r"\{.*\}", resp_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    tool_name = parsed.get("tool")
                    tool_args = parsed.get("args", {})
            except Exception:
                pass

            if not tool_name:
                err_msg = "Error: Could not parse tool call. Reply with JSON: {\"tool\": \"...\", \"args\": {...}}"
                transcript_lines.append(f"--- Step {step} [Tool Result] ---\n{err_msg}")
                history.append({"role": "user", "content": err_msg})
                continue

            if tool_name == "finish":
                transcript_lines.append(f"--- Step {step} [Finished] ---\n{tool_args.get('message', 'Completed')}")
                break
            elif tool_name == "list_files":
                res = str(tools.list_files(tool_args.get("directory", ".")))
            elif tool_name == "read_file":
                res = tools.read_file(tool_args.get("filepath", ""))
            elif tool_name == "write_file":
                fp = tool_args.get("filepath", "")
                content = tool_args.get("content", "")
                res = tools.write_file(fp, content)
                modified_files.add(fp)
            elif tool_name == "run_tests":
                res = tools.run_tests(tool_args.get("args", ""))
            else:
                res = f"Error: Unknown tool '{tool_name}'"

            transcript_lines.append(f"--- Step {step} [Tool Result] ---\n{res}")
            history.append({"role": "user", "content": f"Tool result:\n{res}"})

        transcript_full = "\n".join(transcript_lines)
        return {
            "agent": self.name,
            "provider": provider_used,
            "model": exact_model,
            "mode": "agent",
            "files_modified": sorted(list(modified_files)),
            "transcript": transcript_full,
        }

    def _call_gemini(self, api_key: str, prompt: str) -> tuple[str, str]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            resp_text = candidates[0]["content"]["parts"][0]["text"] if candidates else ""
            exact_model = data.get("modelVersion", self.model_name)
            return resp_text, exact_model

    def _call_openai(self, api_key: str, prompt: str) -> tuple[str, str]:
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            resp_text = data["choices"][0]["message"]["content"]
            exact_model = data.get("model", self.model_name)
            return resp_text, exact_model

    def _call_anthropic(self, api_key: str, prompt: str) -> tuple[str, str]:
        url = "https://api.anthropic.com/v1/messages"
        payload = {
            "model": self.model_name,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            resp_text = data["content"][0]["text"]
            exact_model = data.get("model", self.model_name)
            return resp_text, exact_model
