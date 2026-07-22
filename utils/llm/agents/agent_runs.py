"""Shared agent-run registry.

An :class:`AgentRun` is the agent analogue of a
:class:`~utils.llm.model_runs.ModelRun`: it is the immutable, benchmarkable combination of a
base :class:`~utils.llm.agents.agent_registry.Agent` plus the exact customization that makes
the agent behave a certain way — its system prompt, its tools, and its SDK harness options.

``agent_run_key`` is the stable, immutable benchmark identity (``<agent_key>-run-variant-XX``);
``slug`` is a unique but renameable human-readable convenience name. Both mirror the model-run
conventions so agent runs are drop-in replacements for model runs: an ``AgentRun`` exposes the
same surface ForecastBench drives (``get_response(prompt) -> str`` plus ``model_run_key``,
``slug``, ``provider``, ``provider_model_id``, ``lab``, ``release_date``, ``options``, ...).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final

from .._identifiers import filename_safe_name, validate_registry_key
from ..lab_registry import Lab
from ..provider_registry import Provider
from . import agent_registry
from .agent_sdk_registry import AgentSDK
from .results import AgentResult
from .tools import ToolSpec, WebSearchSpec, tools_fingerprint

logger = logging.getLogger(__name__)

AGENT_RUN_KEY_SUFFIX_PATTERN: Final[re.Pattern[str]] = re.compile(r"^run-variant-[0-9]{2}$")

# A light, domain-neutral system prompt for the shared "research agent" runs. Task-specific
# instructions (e.g. ForecastBench's forecasting prompt) arrive in the user prompt, exactly
# as they do for model runs.
RESEARCH_AGENT_SYSTEM_PROMPT: Final[str] = (
    "You are a careful research assistant. Use your available tools when they help you "
    "answer more accurately, then give a direct, self-contained final answer."
)


@dataclass(frozen=True, slots=True)
class AgentRun:
    """Concrete agent run: a base agent plus its system prompt, tools, and harness options."""

    # Immutable benchmark identifier for this exact agent-plus-config run.
    agent_run_key: str
    # Human-readable convenience identifier for lookup and display; unique but renameable.
    slug: str
    # Canonical base agent supplying the agent SDK route and underlying model.
    agent: agent_registry.Agent
    # System prompt / instructions for the agent. None uses the SDK default.
    system_prompt: str | None = None
    # Tools the agent may use, as provider-neutral specs.
    tools: tuple[ToolSpec, ...] = ()
    # SDK harness options (e.g. max_turns, temperature, reasoning effort, permission_mode).
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate agent-run metadata."""
        _validate_agent_run_key(self.agent_run_key, agent_key=self.agent.agent_key)
        validate_registry_key(self.slug, field_name="AgentRun slug")
        for tool in self.tools:
            if not isinstance(tool, ToolSpec):
                raise TypeError(f"AgentRun tools must be ToolSpec instances, got {tool!r}")

    # -- Agent-native identity -------------------------------------------------------------

    @property
    def agent_key(self) -> str:
        """Return the base agent key."""
        return self.agent.agent_key

    @property
    def agent_sdk(self) -> AgentSDK:
        """Return the external agent SDK that runs this agent."""
        return self.agent.agent_sdk

    # -- ModelRun-compatible surface (drop-in for ForecastBench) ---------------------------

    @property
    def model_run_key(self) -> str:
        """Return the immutable run key under the model-run attribute name (drop-in alias)."""
        return self.agent_run_key

    @property
    def display_name(self) -> str:
        """Return the display name for leaderboards and reports."""
        return self.model_key

    @property
    def filename_safe_name(self) -> str:
        """Return a filename-safe agent-run name."""
        return filename_safe_name(self.agent_run_key, field_name="AgentRun agent_run_key")

    @property
    def model(self) -> Any:
        """Return the underlying base model."""
        return self.agent.model

    @property
    def model_key(self) -> str:
        """Return the underlying base model key."""
        return self.agent.model.model_key

    @property
    def provider_model_id(self) -> str:
        """Return the provider API model identifier."""
        return self.agent.provider_model_id

    @property
    def lab(self) -> Lab:
        """Return the model-making lab."""
        return self.agent.lab

    @property
    def provider(self) -> Provider:
        """Return the API provider route."""
        return self.agent.provider

    @property
    def release_date(self) -> date:
        """Return the underlying model release date."""
        return self.agent.release_date

    def __repr__(self) -> str:
        """Return a concise agent-run representation."""
        tool_kinds = [tool.kind for tool in self.tools]
        return f"<AgentRun {self.agent_run_key} ({self.provider_model_id}) tools={tool_kinds}>"

    # -- Execution -------------------------------------------------------------------------

    def run(self, prompt: str, **option_overrides: Any) -> AgentResult:
        """Run the agent for one prompt and return the full normalized result."""
        merged_options = deepcopy(self.options)
        merged_options.update(deepcopy(option_overrides))
        logger.info(
            "Running agent sdk=%s provider_model_id=%s tools=%s options=%s",
            self.agent_sdk.name,
            self.provider_model_id,
            [tool.kind for tool in self.tools],
            merged_options,
        )
        return agent_registry.run_agent(
            base_agent=self.agent,
            system_prompt=self.system_prompt,
            prompt=prompt,
            tools=self.tools,
            options=merged_options,
        )

    def get_response(self, prompt: str, **option_overrides: Any) -> str:
        """Run the agent and return only its final text (drop-in for ModelRun.get_response)."""
        return self.run(prompt, **option_overrides).final_text

    def with_transcript(self, transcript: Any, context: dict[str, Any] | None = None) -> Any:
        """Return a recording wrapper that logs each call's trajectory to ``transcript``.

        The wrapper presents the same surface as this run, so it can be used anywhere an
        ``AgentRun`` (or ``ModelRun``) is expected, including ForecastBench's runner.
        """
        from .transcript import RecordingAgentRun

        return RecordingAgentRun(self, transcript, context=context)


def _validate_agent_run_key(agent_run_key: str, *, agent_key: str) -> None:
    """Reject invalid agent-run keys."""
    validate_registry_key(agent_run_key, field_name="AgentRun agent_run_key")
    expected_prefix = f"{agent_key}-"
    if not agent_run_key.startswith(expected_prefix):
        raise ValueError(
            "AgentRun agent_run_key must be the agent_key followed by '-run-variant-XX'. "
            f"Expected agent_key: {agent_key}"
        )
    suffix = agent_run_key[len(expected_prefix) :]
    if AGENT_RUN_KEY_SUFFIX_PATTERN.fullmatch(suffix) is None:
        raise ValueError(
            "AgentRun agent_run_key must be the agent_key followed by '-run-variant-XX'. "
            f"Invalid suffix: {suffix}"
        )


def _agent_run_fingerprint(run: AgentRun) -> str:
    """Return a stable fingerprint for an agent's key plus its full run configuration."""
    try:
        payload = json.dumps(
            {
                "agent_key": run.agent_key,
                "system_prompt": run.system_prompt,
                "tools": tools_fingerprint(run.tools),
                "options": run.options,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    except TypeError as exc:
        raise TypeError("AgentRun options must be JSON-serializable for fingerprinting") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _agent_run(
    *,
    agent_run_key: str,
    slug: str,
    agent_key: str,
    system_prompt: str | None = None,
    tools: Sequence[ToolSpec] = (),
    options: dict[str, Any] | None = None,
) -> AgentRun:
    """Create an agent run from a canonical base-agent key."""
    try:
        base_agent = agent_registry.AGENTS_BY_KEY[agent_key]
    except KeyError as exc:
        raise ValueError(f"Unknown agent_key {agent_key}") from exc
    return AgentRun(
        agent_run_key=agent_run_key,
        slug=slug,
        agent=base_agent,
        system_prompt=system_prompt,
        tools=tuple(tools),
        options=deepcopy(options) if options is not None else {},
    )


def _validate_unique_agent_runs(runs: Sequence[AgentRun]) -> None:
    """Reject duplicate agent-run keys, slugs, and semantic fingerprints."""
    seen_keys: set[str] = set()
    seen_slugs: set[str] = set()
    seen_fingerprints: set[str] = set()
    for run in runs:
        if run.agent_run_key in seen_keys:
            raise ValueError(f"Duplicate agent_run_key: {run.agent_run_key}")
        seen_keys.add(run.agent_run_key)
        if run.slug in seen_slugs:
            raise ValueError(f"Duplicate agent-run slug: {run.slug}")
        seen_slugs.add(run.slug)
        fingerprint = _agent_run_fingerprint(run)
        if fingerprint in seen_fingerprints:
            raise ValueError(
                f"Duplicate agent-run fingerprint for agent/config: {run.agent_run_key}"
            )
        seen_fingerprints.add(fingerprint)


def create_agent_runs_list(runs: Sequence[AgentRun]) -> list[AgentRun]:
    """Create a validated agent-run registry list."""
    _validate_unique_agent_runs(runs)
    return list(runs)


# A hosted web-search tool shared by the "research agent" runs below.
_WEB_SEARCH = WebSearchSpec()


CLAUDE_AGENT_RUNS: list[AgentRun] = [
    _agent_run(
        agent_run_key="claude-haiku-4-5-20251001-claude-agent-sdk-run-variant-01",
        slug="claude-haiku-4-5-agent-web-search",
        agent_key="claude-haiku-4-5-20251001-claude-agent-sdk",
        system_prompt=RESEARCH_AGENT_SYSTEM_PROMPT,
        tools=(_WEB_SEARCH,),
        options={"max_turns": 12},
    ),
    _agent_run(
        agent_run_key="claude-opus-4-8-claude-agent-sdk-run-variant-01",
        slug="claude-opus-4-8-agent-web-search",
        agent_key="claude-opus-4-8-claude-agent-sdk",
        system_prompt=RESEARCH_AGENT_SYSTEM_PROMPT,
        tools=(_WEB_SEARCH,),
        options={"max_turns": 16},
    ),
    _agent_run(
        agent_run_key="claude-sonnet-5-claude-agent-sdk-run-variant-01",
        slug="claude-sonnet-5-agent-web-search",
        agent_key="claude-sonnet-5-claude-agent-sdk",
        system_prompt=RESEARCH_AGENT_SYSTEM_PROMPT,
        tools=(_WEB_SEARCH,),
        options={"max_turns": 16},
    ),
]


OPENAI_AGENT_RUNS: list[AgentRun] = [
    _agent_run(
        agent_run_key="gpt-5-mini-2025-08-07-openai-agents-sdk-run-variant-01",
        slug="gpt-5-mini-agent-web-search",
        agent_key="gpt-5-mini-2025-08-07-openai-agents-sdk",
        system_prompt=RESEARCH_AGENT_SYSTEM_PROMPT,
        tools=(_WEB_SEARCH,),
        options={"max_turns": 12},
    ),
    _agent_run(
        agent_run_key="gpt-5.5-2026-04-23-openai-agents-sdk-run-variant-01",
        slug="gpt-5.5-agent-web-search",
        agent_key="gpt-5.5-2026-04-23-openai-agents-sdk",
        system_prompt=RESEARCH_AGENT_SYSTEM_PROMPT,
        tools=(_WEB_SEARCH,),
        options={"max_turns": 16},
    ),
]


AGENT_RUNS: list[AgentRun] = create_agent_runs_list([*CLAUDE_AGENT_RUNS, *OPENAI_AGENT_RUNS])
AGENT_RUNS_BY_KEY: dict[str, AgentRun] = {run.agent_run_key: run for run in AGENT_RUNS}
AGENT_RUNS_BY_SLUG: dict[str, AgentRun] = {run.slug: run for run in AGENT_RUNS}

# AGENT_RUNS is historical. ACTIVE_AGENT_RUNS is the current live-callable subset.
ACTIVE_AGENT_RUNS: list[AgentRun] = [run for run in AGENT_RUNS if run.agent.active]
ACTIVE_AGENT_RUNS_BY_KEY: dict[str, AgentRun] = {
    run.agent_run_key: run for run in ACTIVE_AGENT_RUNS
}
ACTIVE_AGENT_RUNS_BY_SLUG: dict[str, AgentRun] = {run.slug: run for run in ACTIVE_AGENT_RUNS}


def get_agent_run(agent_run_key: str) -> AgentRun:
    """Return a shared agent run by immutable key. Prefer this for durable references."""
    try:
        return AGENT_RUNS_BY_KEY[agent_run_key]
    except KeyError as exc:
        available = ", ".join(sorted(AGENT_RUNS_BY_KEY))
        raise KeyError(f"Unknown agent_run_key {agent_run_key}. Available: {available}") from exc


def get_agent_run_by_slug(slug: str) -> AgentRun:
    """Return a shared agent run by human-readable slug. Convenience lookup only."""
    try:
        return AGENT_RUNS_BY_SLUG[slug]
    except KeyError as exc:
        available = ", ".join(sorted(AGENT_RUNS_BY_SLUG))
        raise KeyError(f"Unknown agent-run slug {slug}. Available: {available}") from exc


def _get_selectable_agent_run(agent_run_key: str, *, active_only: bool) -> AgentRun:
    """Return an agent run from the requested selection scope."""
    if not active_only:
        return get_agent_run(agent_run_key)
    try:
        return ACTIVE_AGENT_RUNS_BY_KEY[agent_run_key]
    except KeyError as exc:
        if agent_run_key in AGENT_RUNS_BY_KEY:
            raise KeyError(
                f"Inactive agent_run_key {agent_run_key}. "
                "Pass active_only=False to select_agent_runs() for historical runs."
            ) from None
        available = ", ".join(sorted(ACTIVE_AGENT_RUNS_BY_KEY))
        raise KeyError(
            f"Unknown active agent_run_key {agent_run_key}. Available: {available}"
        ) from exc


def select_agent_runs(
    agent_run_keys: Sequence[str],
    *,
    active_only: bool = True,
) -> list[AgentRun]:
    """Return active agent runs in the requested order unless historical runs are allowed."""
    return [
        _get_selectable_agent_run(agent_run_key, active_only=active_only)
        for agent_run_key in agent_run_keys
    ]
