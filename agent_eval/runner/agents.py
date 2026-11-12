"""
Agent runner interfaces and implementations for agent evaluation.

Supports:
- ImportAgentRunner: loads hand-crafted or pre-generated solutions from disk/diffs.
- DirectoryOverlayRunner: applies a local folder overlay (used for reference/known-bad testing).
- LLMAgentRunner: calls an LLM provider (OpenAI, Anthropic, or Gemini) using environment keys.
"""

import json
import os
import re
import shutil
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


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
                copied_files.append(str(rel_path))

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

        # If it is a diff/patch file or JSON run record
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
        """Parses git-style unified diff blocks and writes replaced files."""
        # Simple file extractor from diff or file fences
        file_blocks = re.findall(
            r"diff --git a/(.*?) b/(.*?)\n.*?(?=(?:diff --git|\Z))",
            patch_text,
            re.DOTALL,
        )
        if not file_blocks:
            # Check for file path blocks like: +++ b/path/to/file
            file_matches = re.finditer(r"\+\+\+ b/(.*?)\n(.*)", patch_text, re.DOTALL)
            for m in file_matches:
                rel_path = m.group(1).strip()
                dest = workspace_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)


class LLMAgentRunner(AgentRunner):
    """
    Executes an LLM API call using credentials from environment variables.
    Checks GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.
    """

    def __init__(self, model_name: str | None = None, name: str = "llm_agent"):
        super().__init__(name=name)
        self.model_name = model_name or os.environ.get("AGENT_EVAL_MODEL", "gemini-2.5-pro")

    def solve(self, prompt: str, workspace_dir: Path) -> dict[str, Any]:
        # Identify available API key
        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

        if not (gemini_key or openai_key or anthropic_key):
            raise ValueError(
                "No LLM API key detected. Please set one of GEMINI_API_KEY, "
                "OPENAI_API_KEY, or ANTHROPIC_API_KEY to run live LLMAgentRunner."
            )

        # Inspect workspace files
        context_files: dict[str, str] = {}
        for p in workspace_dir.rglob("*.py"):
            if "__pycache__" not in p.parts:
                context_files[str(p.relative_to(workspace_dir))] = p.read_text(encoding="utf-8")

        for p in workspace_dir.glob("*.md"):
            context_files[str(p.relative_to(workspace_dir))] = p.read_text(encoding="utf-8")

        prompt_payload = (
            f"You are an expert quantitative software engineering agent.\n"
            f"Here is your assignment:\n\n{prompt}\n\n"
            f"Existing repository files:\n"
        )
        for rel_name, content in context_files.items():
            prompt_payload += f"\n--- {rel_name} ---\n{content}\n"

        prompt_payload += (
            "\nOutput all updated or created files in standard markdown code fences, "
            "with the relative filepath on the first line after ```python or ```markdown.\n"
            "Example:\n```python filepath/to/file.py\n# code\n```\n"
        )

        response_text = ""
        provider_used = ""

        if gemini_key:
            provider_used = "gemini"
            response_text = self._call_gemini(gemini_key, prompt_payload)
        elif openai_key:
            provider_used = "openai"
            response_text = self._call_openai(openai_key, prompt_payload)
        elif anthropic_key:
            provider_used = "anthropic"
            response_text = self._call_anthropic(anthropic_key, prompt_payload)

        # Parse generated files from response
        modified_files = self._apply_code_blocks(response_text, workspace_dir)

        return {
            "agent": self.name,
            "provider": provider_used,
            "model": self.model_name,
            "files_modified": modified_files,
            "transcript": response_text,
        }

    def _call_gemini(self, api_key: str, prompt: str) -> str:
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
            if candidates:
                return candidates[0]["content"]["parts"][0]["text"]
            return ""

    def _call_openai(self, api_key: str, prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.model_name if self.model_name.startswith("gpt") else "gpt-4o",
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
            return data["choices"][0]["message"]["content"]

    def _call_anthropic(self, api_key: str, prompt: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        payload = {
            "model": self.model_name if "claude" in self.model_name else "claude-3-5-sonnet-20241022",
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
            return data["content"][0]["text"]

    def _apply_code_blocks(self, text: str, workspace_dir: Path) -> list[str]:
        pattern = re.compile(r"```(?:\w+)?\s+([^\n\r]+)\n(.*?)```", re.DOTALL)
        modified = []
        for match in pattern.finditer(text):
            filename_header = match.group(1).strip()
            code_content = match.group(2)
            # Clean filename header (strip comments or backticks)
            filepath_clean = re.sub(r"^[#\s/]+", "", filename_header).strip()
            # If the header contains spaces, take the first token
            filepath_tokens = filepath_clean.split()
            if filepath_tokens:
                filepath_str = filepath_tokens[0]
                target = workspace_dir / filepath_str
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(code_content, encoding="utf-8")
                modified.append(filepath_str)
        return modified
