"""Central registry of base agents.

A base :class:`Agent` is the agent analogue of a :class:`~utils.llm.model_registry.Model`:
it is the reusable, provider-callable combination of an external agent SDK
(:class:`~utils.llm.agents.agent_sdk_registry.AgentSDK`) and an underlying base model. It
carries routing and metadata only; the per-run customization that makes an agent behave a
certain way (system prompt, tools, harness options) lives on
:class:`~utils.llm.agents.agent_runs.AgentRun`, exactly as provider options live on
``ModelRun`` rather than ``Model``.

Adding a base agent:

1. Ensure the underlying model exists in ``utils.llm.model_registry`` (add it there first if
   not). The model's provider route must match the agent SDK's ``provider_key``.
2. Add an entry to the SDK-specific list below via ``claude_agent`` / ``openai_agent``.
3. Declare benchmarkable configs in ``agent_runs.py`` with explicit ``agent_run_key`` values.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final, Type

from .. import model_registry
from .._identifiers import filename_safe_name, validate_registry_key
from ..lab_registry import Lab
from ..provider_registry import PROVIDERS, Provider
from .agent_sdk_registry import AGENT_SDKS, AgentSDK
from .providers.anthropic_agent import ClaudeAgentSDKProvider
from .providers.base import BaseAgentProvider
from .providers.openai_agent import OpenAIAgentsProvider
from .results import AgentResult
from .tools import ToolSpec

# Mapping from agent SDKs to their provider classes.
_AGENT_SDK_TO_PROVIDER_CLASS: Final[dict[AgentSDK, Type[BaseAgentProvider]]] = {
    AGENT_SDKS["Claude Agent SDK"]: ClaudeAgentSDKProvider,
    AGENT_SDKS["OpenAI Agents SDK"]: OpenAIAgentsProvider,
}

# Cache of provider instances keyed by (class, api_key) so key changes rebuild them.
_AGENT_PROVIDER_INSTANCES: dict[tuple[Type[BaseAgentProvider], str], BaseAgentProvider] = {}


@dataclass(frozen=True, slots=True)
class Agent:
    """Canonical base agent: an agent SDK bound to an underlying base model."""

    # Stable registry key for the base agent. Agent runs reference this in their keys.
    agent_key: str
    # The external agent framework that drives this agent's loop and tool execution.
    agent_sdk: AgentSDK
    # Canonical base model registry entry supplying routing and release metadata.
    model: model_registry.Model

    def __post_init__(self) -> None:
        """Validate the agent declaration against its SDK and model."""
        validate_registry_key(self.agent_key, field_name="Agent agent_key")
        expected_provider = PROVIDERS[self.agent_sdk.provider_key]
        if self.model.provider != expected_provider:
            raise ValueError(
                f"Agent {self.agent_key}: {self.agent_sdk.name} requires a "
                f"{expected_provider.name} model, but model {self.model.model_key} routes "
                f"through {self.model.provider.name}."
            )

    @property
    def provider(self) -> Provider:
        """Return the underlying model's API provider route."""
        return self.model.provider

    @property
    def provider_model_id(self) -> str:
        """Return the exact model id the agent SDK runs."""
        return self.model.provider_model_id

    @property
    def lab(self) -> Lab:
        """Return the model-making lab."""
        return self.model.lab

    @property
    def release_date(self) -> date:
        """Return the underlying model's release date."""
        return self.model.release_date

    @property
    def active(self) -> bool:
        """Return whether the underlying base model is active."""
        return self.model.active

    @property
    def filename_safe_name(self) -> str:
        """Return a filename-safe agent name."""
        return filename_safe_name(self.agent_key, field_name="Agent agent_key")


def agent(*, agent_key: str, agent_sdk_name: str, model_key: str) -> Agent:
    """Create a base agent from an agent SDK name and a base model key."""
    try:
        agent_sdk = AGENT_SDKS[agent_sdk_name]
    except KeyError as exc:
        available = ", ".join(sorted(AGENT_SDKS))
        raise ValueError(f"Unknown agent SDK {agent_sdk_name}. Available: {available}") from exc
    try:
        model = model_registry.MODELS_BY_KEY[model_key]
    except KeyError as exc:
        raise ValueError(f"Unknown model_key {model_key}") from exc
    return Agent(agent_key=agent_key, agent_sdk=agent_sdk, model=model)


def claude_agent(*, model_key: str, agent_key: str | None = None) -> Agent:
    """Create a Claude Agent SDK base agent for an Anthropic base model."""
    return agent(
        agent_key=agent_key or f"{model_key}-claude-agent-sdk",
        agent_sdk_name="Claude Agent SDK",
        model_key=model_key,
    )


def openai_agent(*, model_key: str, agent_key: str | None = None) -> Agent:
    """Create an OpenAI Agents SDK base agent for an OpenAI base model."""
    return agent(
        agent_key=agent_key or f"{model_key}-openai-agents-sdk",
        agent_sdk_name="OpenAI Agents SDK",
        model_key=model_key,
    )


# Claude Agent SDK base agents.
CLAUDE_AGENTS: Final[list[Agent]] = [
    claude_agent(model_key="claude-haiku-4-5-20251001"),
    claude_agent(model_key="claude-opus-4-8"),
    claude_agent(model_key="claude-sonnet-5"),
]

# OpenAI Agents SDK base agents.
OPENAI_AGENTS: Final[list[Agent]] = [
    openai_agent(model_key="gpt-5-mini-2025-08-07"),
    openai_agent(model_key="gpt-5.5-2026-04-23"),
]


def _validate_unique_agent_keys(agents: Sequence[Agent]) -> None:
    """Reject duplicate agent keys in an agent registry list."""
    seen: set[str] = set()
    for base_agent in agents:
        if base_agent.agent_key in seen:
            raise ValueError(f"Duplicate agent_key: {base_agent.agent_key}")
        seen.add(base_agent.agent_key)


def create_agents_list(agents: Sequence[Agent]) -> list[Agent]:
    """Create a validated base-agent registry list."""
    _validate_unique_agent_keys(agents)
    return list(agents)


AGENTS: Final[list[Agent]] = create_agents_list([*CLAUDE_AGENTS, *OPENAI_AGENTS])
AGENTS_BY_KEY: Final[dict[str, Agent]] = {base.agent_key: base for base in AGENTS}


def _agent_provider_instance(base_agent: Agent) -> BaseAgentProvider:
    """Return a cached provider instance for a base agent, authenticated from the registry."""
    try:
        provider_cls = _AGENT_SDK_TO_PROVIDER_CLASS[base_agent.agent_sdk]
    except KeyError as exc:
        raise ValueError(f"Unsupported agent SDK: {base_agent.agent_sdk.name}") from exc

    api_key = model_registry.configured_api_key_for_provider(base_agent.provider)
    if not api_key:
        raise ValueError(
            f"API key not configured for {base_agent.provider.name} (required by "
            f"{base_agent.agent_sdk.name}). Call configure_api_keys() or "
            "configure_api_keys(from_gcp=True)."
        )

    cache_key = (provider_cls, api_key)
    instance = _AGENT_PROVIDER_INSTANCES.get(cache_key)
    if instance is None:
        instance = provider_cls(api_key=api_key)
        _AGENT_PROVIDER_INSTANCES[cache_key] = instance
    return instance


def run_agent(
    *,
    base_agent: Agent,
    system_prompt: str | None,
    prompt: str,
    tools: Sequence[ToolSpec],
    options: dict[str, Any],
) -> AgentResult:
    """Run a base agent for one prompt through its configured SDK provider."""
    provider = _agent_provider_instance(base_agent)
    return provider.run(
        model_id=base_agent.provider_model_id,
        system_prompt=system_prompt,
        prompt=prompt,
        tools=tools,
        options=options,
    )
