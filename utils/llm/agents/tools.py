"""Provider-neutral tool declarations for agent runs.

Agents are defined largely by the tools they may use. These specs describe a tool once,
in an SDK-agnostic way; each agent provider module (``anthropic_agent``, ``openai_agent``)
translates the specs into its own SDK's native tool objects. Three tool categories are
supported, matching what both the Claude Agent SDK and the OpenAI Agents SDK expose:

* hosted web tools (:class:`WebSearchSpec`, :class:`WebFetchSpec`),
* in-process Python function tools (:class:`FunctionToolSpec`), and
* Model Context Protocol servers (:class:`MCPStdioServerSpec`, :class:`MCPHttpServerSpec`).

Every spec exposes :meth:`ToolSpec.fingerprint`, a JSON-serializable descriptor of the
tool's *configuration* (never the live Python handler) used for agent-run uniqueness
checks and tests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """Base class for provider-neutral tool declarations."""

    #: Stable discriminator for the tool category. Overridden by each subclass.
    kind: str = field(init=False, default="tool")

    def fingerprint(self) -> dict[str, Any]:
        """Return a JSON-serializable descriptor of this tool's configuration."""
        raise NotImplementedError


@dataclass(frozen=True)
class WebSearchSpec(ToolSpec):
    """Hosted web-search tool.

    Maps to the Claude Agent SDK ``WebSearch`` built-in tool and to the OpenAI Agents SDK
    ``WebSearchTool``. Options that a given SDK does not support are ignored by that SDK's
    translator.
    """

    kind: str = field(init=False, default="web_search")
    # Cap on the number of searches per run (Claude ``max_uses``); None leaves it unset.
    max_uses: int | None = None
    # Optional domain allow/deny lists (Claude ``allowed_domains``/``blocked_domains``;
    # OpenAI ``filters``). None leaves them unset.
    allowed_domains: tuple[str, ...] | None = None
    blocked_domains: tuple[str, ...] | None = None
    # OpenAI ``search_context_size``: "low" | "medium" | "high".
    search_context_size: str = "medium"

    def fingerprint(self) -> dict[str, Any]:
        """Return a JSON-serializable descriptor of this tool's configuration."""
        return {
            "kind": self.kind,
            "max_uses": self.max_uses,
            "allowed_domains": list(self.allowed_domains) if self.allowed_domains else None,
            "blocked_domains": list(self.blocked_domains) if self.blocked_domains else None,
            "search_context_size": self.search_context_size,
        }


@dataclass(frozen=True)
class WebFetchSpec(ToolSpec):
    """Hosted web-fetch tool (fetch and read a specific URL).

    Maps to the Claude Agent SDK ``WebFetch`` built-in tool. The OpenAI Agents SDK has no
    direct equivalent; its translator ignores this spec.
    """

    kind: str = field(init=False, default="web_fetch")
    max_uses: int | None = None

    def fingerprint(self) -> dict[str, Any]:
        """Return a JSON-serializable descriptor of this tool's configuration."""
        return {"kind": self.kind, "max_uses": self.max_uses}


@dataclass(frozen=True)
class FunctionToolSpec(ToolSpec):
    """In-process Python function exposed to the agent as a callable tool.

    ``parameters`` must be a JSON Schema object (``{"type": "object", "properties": {...}}``);
    both SDKs accept a full JSON Schema for the tool inputs. ``handler`` is a plain Python
    callable (sync or async) invoked as ``handler(**arguments)``; its return value is
    stringified into the tool result.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Callable[..., Any]
    kind: str = field(init=False, default="function")

    def fingerprint(self) -> dict[str, Any]:
        """Return a JSON-serializable descriptor of this tool's configuration.

        The live ``handler`` is intentionally excluded; a function tool's identity is its
        name, description, and input schema.
        """
        return {
            "kind": self.kind,
            "name": self.name,
            "description": self.description,
            "parameters": _jsonify(self.parameters),
        }


@dataclass(frozen=True)
class MCPStdioServerSpec(ToolSpec):
    """A Model Context Protocol server launched as a local stdio subprocess."""

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    # Optional subset of tool names to allow from this server; None allows all.
    tool_names: tuple[str, ...] | None = None
    kind: str = field(init=False, default="mcp_stdio")

    def fingerprint(self) -> dict[str, Any]:
        """Return a JSON-serializable descriptor of this tool's configuration."""
        return {
            "kind": self.kind,
            "name": self.name,
            "command": self.command,
            "args": list(self.args),
            "env": dict(self.env) if self.env else None,
            "tool_names": list(self.tool_names) if self.tool_names else None,
        }


@dataclass(frozen=True)
class MCPHttpServerSpec(ToolSpec):
    """A remote Model Context Protocol server reached over (streamable) HTTP."""

    name: str
    url: str
    headers: Mapping[str, str] | None = None
    tool_names: tuple[str, ...] | None = None
    # Claude/OpenAI hosted-MCP approval policy: "never" runs tools without approval.
    require_approval: str = "never"
    kind: str = field(init=False, default="mcp_http")

    def fingerprint(self) -> dict[str, Any]:
        """Return a JSON-serializable descriptor of this tool's configuration."""
        return {
            "kind": self.kind,
            "name": self.name,
            "url": self.url,
            "headers": dict(self.headers) if self.headers else None,
            "tool_names": list(self.tool_names) if self.tool_names else None,
            "require_approval": self.require_approval,
        }


def tools_fingerprint(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    """Return an ordered list of tool fingerprints for an agent-run fingerprint."""
    return [tool.fingerprint() for tool in tools]


def _jsonify(value: Any) -> Any:
    """Return a JSON-safe copy of nested mappings/sequences for fingerprinting."""
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value
