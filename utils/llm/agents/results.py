"""Normalized results returned by agent providers.

Each agent SDK reports its run differently. Provider modules translate their native output
into these provider-neutral structures so callers, transcripts, and tests see one shape
regardless of which SDK ran the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Step kinds, normalized across SDKs.
STEP_TEXT = "text"
STEP_REASONING = "reasoning"
STEP_TOOL_USE = "tool_use"
STEP_TOOL_RESULT = "tool_result"
STEP_SYSTEM = "system"
STEP_OTHER = "other"


@dataclass(frozen=True)
class AgentStep:
    """One normalized step in an agent's trajectory."""

    # One of the STEP_* constants above.
    kind: str
    # Assistant/reasoning text, when the step carries text.
    text: str | None = None
    # Tool name for tool_use / tool_result steps.
    tool_name: str | None = None
    # Tool input arguments for tool_use steps (JSON-serializable where possible).
    tool_input: Any = None
    # Tool output payload for tool_result steps.
    tool_output: Any = None
    # The raw SDK message/item type name, kept for debugging.
    raw_type: str | None = None


@dataclass
class AgentResult:
    """Normalized result of a single agent run."""

    # The agent's final assistant text. This is what ``get_response`` returns.
    final_text: str
    # Ordered trajectory of normalized steps (assistant text, tool calls, results, ...).
    steps: list[AgentStep] = field(default_factory=list)
    # Provider-reported token usage, when available.
    usage: dict[str, Any] | None = None
    # Provider-reported cost in USD, when available (Claude Agent SDK reports this).
    cost_usd: float | None = None
    # Number of agent-loop turns, when available.
    num_turns: int | None = None
    # True when the SDK reported the run ended in an error state.
    is_error: bool = False
    # Which agent SDK produced this result (human-readable name).
    agent_sdk_name: str | None = None
    # The underlying model id the SDK ran.
    model_id: str | None = None
    # The raw SDK result object, kept for debugging; not serialized.
    raw: Any = None

    def tool_use_steps(self) -> list[AgentStep]:
        """Return only the tool-invocation steps in this run."""
        return [step for step in self.steps if step.kind == STEP_TOOL_USE]
