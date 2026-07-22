"""Capture agent-run trajectories to Markdown and JSONL transcripts.

Model runs return only their final text, but an agent run also produces a trajectory of tool
calls and intermediate steps. :class:`AgentRunTranscript` records that trajectory (plus token
usage and cost) to human-readable Markdown and machine-readable JSONL. :class:`RecordingAgentRun`
wraps an :class:`~utils.llm.agents.agent_runs.AgentRun` so every call is recorded while still
presenting the exact ModelRun surface ForecastBench drives, making transcript capture a drop-in
concern.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any

from .agent_runs import AgentRun
from .results import AgentResult, AgentStep


class AgentRunTranscript:
    """Thread-safe writer for agent-run trajectories (Markdown + JSONL)."""

    def __init__(self, local_filename: str | Path) -> None:
        """Create fresh transcript files for this run."""
        self.base_filename = Path(local_filename)
        self.markdown_filename = Path(f"{self.base_filename}.agent-calls.md")
        self.jsonl_filename = Path(f"{self.base_filename}.agent-calls.jsonl")
        self.local_filename = self.markdown_filename
        self._write_text(self.markdown_filename, "# Agent Call Transcript\n")
        self._write_text(self.jsonl_filename, "")
        self._lock = Lock()
        self._next_call_index = 1

    @staticmethod
    def _write_text(local_filename: Path, text: str) -> None:
        local_filename.parent.mkdir(parents=True, exist_ok=True)
        local_filename.write_text(text, encoding="utf-8")

    @staticmethod
    def _append_text(local_filename: Path, text: str) -> None:
        local_filename.parent.mkdir(parents=True, exist_ok=True)
        with local_filename.open("a", encoding="utf-8") as handle:
            handle.write(text)

    @staticmethod
    def _fenced(value: str | None) -> str:
        value = value or ""
        fence = "```"
        while fence in value:
            fence += "`"
        return f"{fence}text\n{value}\n{fence}"

    def record(
        self,
        agent_run: AgentRun,
        prompt: str,
        result: AgentResult | None = None,
        error: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Append one completed agent call (prompt, steps, result) to the transcript."""
        context = context or {}
        with self._lock:
            call_index = self._next_call_index
            self._next_call_index += 1
            self._append_text(
                self.markdown_filename,
                self._render_markdown(call_index, agent_run, prompt, result, error, context),
            )
            record = {
                "call_index": call_index,
                "agent_run_key": agent_run.agent_run_key,
                "agent_run_slug": agent_run.slug,
                "agent_sdk": agent_run.agent_sdk.name,
                "provider": agent_run.provider.name,
                "lab": agent_run.lab.name,
                "provider_model_id": agent_run.provider_model_id,
                "tools": [tool.fingerprint() for tool in agent_run.tools],
                "system_prompt": agent_run.system_prompt,
                "prompt": prompt,
                "final_text": result.final_text if result else None,
                "steps": [_step_to_dict(step) for step in result.steps] if result else [],
                "usage": result.usage if result else None,
                "cost_usd": result.cost_usd if result else None,
                "num_turns": result.num_turns if result else None,
                "is_error": result.is_error if result else True,
                "error": error,
                **context,
            }
            self._append_text(
                self.jsonl_filename,
                f"{json.dumps(record, ensure_ascii=False, default=str)}\n",
            )

    def _render_markdown(
        self,
        call_index: int,
        agent_run: AgentRun,
        prompt: str,
        result: AgentResult | None,
        error: str | None,
        context: dict[str, Any],
    ) -> str:
        label = context.get("role", "agent")
        section = [
            "",
            f"## Call {call_index}: {label}",
            "",
            f"- Agent SDK: {agent_run.agent_sdk.name}",
            f"- Agent run key: {agent_run.agent_run_key}",
            f"- Agent run slug: {agent_run.slug}",
            f"- Provider model ID: {agent_run.provider_model_id}",
            f"- Tools: {[tool.kind for tool in agent_run.tools]}",
        ]
        for key, value in context.items():
            if key == "role":
                continue
            section.append(f"- {key}: {value}")
        if result is not None:
            section.append(f"- Turns: {result.num_turns}  Cost USD: {result.cost_usd}")
        section += ["", "### Prompt", "", self._fenced(prompt)]
        if result is not None:
            section += ["", "### Steps", "", self._render_steps(result.steps)]
            section += ["", "### Final answer", "", self._fenced(result.final_text)]
        if error is not None:
            section += ["", "### Error", "", self._fenced(error)]
        return "\n".join(section) + "\n"

    def _render_steps(self, steps: list[AgentStep]) -> str:
        if not steps:
            return "_(no steps recorded)_"
        lines = []
        for index, step in enumerate(steps, start=1):
            if step.kind == "tool_use":
                detail = f"tool `{step.tool_name}` input={_short(step.tool_input)}"
            elif step.kind == "tool_result":
                detail = f"tool result={_short(step.tool_output)}"
            else:
                detail = _short(step.text)
            lines.append(f"{index}. **{step.kind}** — {detail}")
        return "\n".join(lines)


class RecordingAgentRun:
    """Wrap an AgentRun so each call records its full trajectory to a transcript.

    Delegates every other attribute to the wrapped run, so this object presents the same
    surface as an :class:`AgentRun` (and therefore a ``ModelRun``) and can be passed straight
    into ForecastBench's runner.
    """

    def __init__(
        self,
        agent_run: AgentRun,
        transcript: AgentRunTranscript,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Store the wrapped run, transcript sink, and optional recording context."""
        self._agent_run = agent_run
        self._transcript = transcript
        self._context = context or {}

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped agent run."""
        return getattr(self._agent_run, name)

    def run(self, prompt: str, **option_overrides: Any) -> AgentResult:
        """Run the wrapped agent, record the trajectory, and return the full result."""
        try:
            result = self._agent_run.run(prompt, **option_overrides)
        except Exception as exc:
            self._transcript.record(
                self._agent_run,
                prompt,
                result=None,
                error=f"{type(exc).__name__}: {exc}",
                context=self._context,
            )
            raise
        self._transcript.record(self._agent_run, prompt, result=result, context=self._context)
        return result

    def get_response(self, prompt: str, **option_overrides: Any) -> str:
        """Run the wrapped agent, record the trajectory, and return only the final text."""
        return self.run(prompt, **option_overrides).final_text


def _step_to_dict(step: AgentStep) -> dict[str, Any]:
    """Return a JSON-safe dict for one agent step."""
    return asdict(step)


def _short(value: Any, limit: int = 500) -> str:
    """Return a compact single-line preview of a value for Markdown rendering."""
    if value is None:
        return "None"
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"
