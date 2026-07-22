"""Registry of external agent SDKs used to run agents.

An ``AgentSDK`` is the *framework* axis of an agent: the external library that owns
the agent loop, tool execution, and context management (for example the Claude Agent
SDK or the OpenAI Agents SDK). It is deliberately separate from the model API
:class:`~utils.llm.provider_registry.Provider`: an agent SDK drives an underlying
model that is still routed through one of the existing provider routes, and
``provider_key`` records which provider route an SDK's base models must use.

Adding a new agent SDK is intentionally analogous to adding a model provider: add an
entry here, then add a matching provider module under ``agents/providers/`` and wire it
into ``agent_registry._AGENT_SDK_TO_PROVIDER_CLASS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class AgentSDK:
    """Metadata for an external agent framework/SDK."""

    # Human-readable name used in labels and transcripts.
    name: str
    # Stable machine key (snake_case) used in agent keys and filenames.
    key_name: str
    # provider_registry key whose route an SDK's base models must use, and whose
    # configured API key authenticates the SDK (e.g. "Anthropic", "OpenAI").
    provider_key: str


AGENT_SDKS: Final[dict[str, AgentSDK]] = {
    "Claude Agent SDK": AgentSDK(
        name="Claude Agent SDK",
        key_name="claude_agent_sdk",
        provider_key="Anthropic",
    ),
    "OpenAI Agents SDK": AgentSDK(
        name="OpenAI Agents SDK",
        key_name="openai_agents",
        provider_key="OpenAI",
    ),
}
