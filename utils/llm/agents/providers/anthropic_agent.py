"""Agent provider backed by the Claude Agent SDK.

Translates provider-neutral :mod:`~utils.llm.agents.tools` specs into ``ClaudeAgentOptions``
and drives a single-prompt agent run through ``claude_agent_sdk.query``. The underlying
Claude Code CLI executes the agent loop; this module normalizes its streamed messages into
an :class:`~utils.llm.agents.results.AgentResult`.

Runs are made reproducible by default: ambient ``.claude`` settings are not loaded
(``setting_sources=[]``), and the built-in filesystem/exec tools are disallowed unless a
tool spec explicitly asks for them, so the agent only uses the tools it was given.
"""

from __future__ import annotations

import inspect
import json
import tempfile
from collections.abc import Sequence
from typing import Any, Final

from ..results import (
    STEP_REASONING,
    STEP_TEXT,
    STEP_TOOL_RESULT,
    STEP_TOOL_USE,
    AgentResult,
    AgentStep,
)
from ..tools import (
    FunctionToolSpec,
    MCPHttpServerSpec,
    MCPStdioServerSpec,
    ToolSpec,
    WebFetchSpec,
    WebSearchSpec,
)
from .base import BaseAgentProvider

# Built-in Claude Code tools disallowed by default so benchmark agents only use declared
# tools. Web tools are opt-in and removed from this denylist when a spec requests them.
_DEFAULT_DISALLOWED_TOOLS: Final[tuple[str, ...]] = (
    "Bash",
    "BashOutput",
    "KillShell",
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "Glob",
    "Grep",
    "Task",
    "TodoWrite",
    "SlashCommand",
    "WebSearch",
    "WebFetch",
)

# ClaudeAgentOptions fields passed straight through from an agent run's ``options``.
_PASSTHROUGH_OPTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "max_turns",
        "permission_mode",
        "max_thinking_tokens",
        "thinking",
        "effort",
        "betas",
        "max_budget_usd",
        "add_dirs",
        "extra_args",
        "fallback_model",
    }
)

_IN_PROCESS_FUNCTION_SERVER: Final[str] = "fri_functions"


class ClaudeAgentSDKProvider(BaseAgentProvider):
    """Agent provider that runs agents through the Claude Agent SDK."""

    retry_message = "Claude Agent SDK run failed."

    async def _arun(
        self,
        *,
        model_id: str,
        system_prompt: str | None,
        prompt: str,
        tools: Sequence[ToolSpec],
        options: dict[str, Any],
    ) -> AgentResult:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            UserMessage,
            query,
        )

        allowed_tools, mcp_servers, enabled_builtins = self._translate_tools(tools)
        disallowed_tools = [
            name for name in _DEFAULT_DISALLOWED_TOOLS if name not in enabled_builtins
        ]

        agent_options_kwargs: dict[str, Any] = {
            "model": model_id,
            "env": {"ANTHROPIC_API_KEY": self._api_key},
            "allowed_tools": allowed_tools,
            "disallowed_tools": disallowed_tools,
            "mcp_servers": mcp_servers,
            "permission_mode": options.get("permission_mode", "bypassPermissions"),
            "setting_sources": options.get("setting_sources", []),
            "cwd": options.get("cwd", tempfile.gettempdir()),
        }
        if system_prompt is not None:
            agent_options_kwargs["system_prompt"] = system_prompt
        for key in _PASSTHROUGH_OPTION_KEYS:
            if key in options and key not in ("permission_mode",):
                agent_options_kwargs[key] = options[key]

        agent_options = ClaudeAgentOptions(**agent_options_kwargs)

        steps: list[AgentStep] = []
        assistant_text_parts: list[str] = []
        result_message: Any = None

        async for message in query(prompt=prompt, options=agent_options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    step = _assistant_block_to_step(block)
                    if step is None:
                        continue
                    if step.kind == STEP_TEXT and step.text:
                        assistant_text_parts.append(step.text)
                    steps.append(step)
            elif isinstance(message, UserMessage):
                for step in _user_message_tool_results(message):
                    steps.append(step)
            elif isinstance(message, ResultMessage):
                result_message = message

        return _build_result(
            model_id=model_id,
            steps=steps,
            assistant_text_parts=assistant_text_parts,
            result_message=result_message,
        )

    def _translate_tools(
        self, tools: Sequence[ToolSpec]
    ) -> tuple[list[str], dict[str, Any], set[str]]:
        """Return (allowed_tools, mcp_servers, enabled_builtin_names) for the given specs."""
        allowed_tools: list[str] = []
        mcp_servers: dict[str, Any] = {}
        enabled_builtins: set[str] = set()
        function_specs: list[FunctionToolSpec] = []

        for tool in tools:
            if isinstance(tool, WebSearchSpec):
                allowed_tools.append("WebSearch")
                enabled_builtins.add("WebSearch")
            elif isinstance(tool, WebFetchSpec):
                allowed_tools.append("WebFetch")
                enabled_builtins.add("WebFetch")
            elif isinstance(tool, FunctionToolSpec):
                function_specs.append(tool)
            elif isinstance(tool, MCPStdioServerSpec):
                mcp_servers[tool.name] = {
                    "type": "stdio",
                    "command": tool.command,
                    "args": list(tool.args),
                    **({"env": dict(tool.env)} if tool.env else {}),
                }
                allowed_tools.extend(_mcp_allowed_names(tool.name, tool.tool_names))
            elif isinstance(tool, MCPHttpServerSpec):
                mcp_servers[tool.name] = {
                    "type": "http",
                    "url": tool.url,
                    **({"headers": dict(tool.headers)} if tool.headers else {}),
                }
                allowed_tools.extend(_mcp_allowed_names(tool.name, tool.tool_names))
            else:
                raise TypeError(f"Unsupported tool spec for Claude Agent SDK: {tool!r}")

        if function_specs:
            from claude_agent_sdk import create_sdk_mcp_server

            sdk_tools = [_make_sdk_function_tool(spec) for spec in function_specs]
            mcp_servers[_IN_PROCESS_FUNCTION_SERVER] = create_sdk_mcp_server(
                name=_IN_PROCESS_FUNCTION_SERVER,
                version="1.0.0",
                tools=sdk_tools,
            )
            allowed_tools.extend(
                f"mcp__{_IN_PROCESS_FUNCTION_SERVER}__{spec.name}" for spec in function_specs
            )

        return allowed_tools, mcp_servers, enabled_builtins


def _mcp_allowed_names(server_name: str, tool_names: tuple[str, ...] | None) -> list[str]:
    """Return allowed-tool entries for an MCP server (all tools or a named subset)."""
    if not tool_names:
        return [f"mcp__{server_name}"]
    return [f"mcp__{server_name}__{name}" for name in tool_names]


def _make_sdk_function_tool(spec: FunctionToolSpec) -> Any:
    """Wrap a FunctionToolSpec as a Claude Agent SDK in-process tool."""
    from claude_agent_sdk import tool as sdk_tool

    @sdk_tool(spec.name, spec.description, dict(spec.parameters))
    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        outcome = spec.handler(**args)
        if inspect.isawaitable(outcome):
            outcome = await outcome
        return {"content": [{"type": "text", "text": _stringify(outcome)}]}

    return _handler


def _assistant_block_to_step(block: Any) -> AgentStep | None:
    """Normalize one assistant content block into an AgentStep."""
    from claude_agent_sdk import (
        ServerToolUseBlock,
        TextBlock,
        ThinkingBlock,
        ToolResultBlock,
        ToolUseBlock,
    )

    raw_type = type(block).__name__
    if isinstance(block, TextBlock):
        return AgentStep(kind=STEP_TEXT, text=block.text, raw_type=raw_type)
    if isinstance(block, ThinkingBlock):
        return AgentStep(
            kind=STEP_REASONING, text=getattr(block, "thinking", None), raw_type=raw_type
        )
    if isinstance(block, (ToolUseBlock, ServerToolUseBlock)):
        return AgentStep(
            kind=STEP_TOOL_USE,
            tool_name=getattr(block, "name", None),
            tool_input=getattr(block, "input", None),
            raw_type=raw_type,
        )
    if isinstance(block, ToolResultBlock):
        return AgentStep(
            kind=STEP_TOOL_RESULT,
            tool_output=getattr(block, "content", None),
            raw_type=raw_type,
        )
    return AgentStep(kind="other", raw_type=raw_type)


def _user_message_tool_results(message: Any) -> list[AgentStep]:
    """Extract tool-result steps from a UserMessage fed back into the loop."""
    from claude_agent_sdk import ToolResultBlock

    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return []
    steps: list[AgentStep] = []
    for block in content:
        if isinstance(block, ToolResultBlock):
            steps.append(
                AgentStep(
                    kind=STEP_TOOL_RESULT,
                    tool_output=getattr(block, "content", None),
                    raw_type=type(block).__name__,
                )
            )
    return steps


def _build_result(
    *,
    model_id: str,
    steps: list[AgentStep],
    assistant_text_parts: list[str],
    result_message: Any,
) -> AgentResult:
    """Assemble a normalized AgentResult from streamed Claude Agent SDK messages."""
    final_text = ""
    usage = None
    cost_usd = None
    num_turns = None
    is_error = False
    if result_message is not None:
        result_text = getattr(result_message, "result", None)
        if isinstance(result_text, str):
            final_text = result_text
        usage = getattr(result_message, "usage", None)
        cost_usd = getattr(result_message, "total_cost_usd", None)
        num_turns = getattr(result_message, "num_turns", None)
        is_error = bool(getattr(result_message, "is_error", False))
    if not final_text:
        final_text = "".join(assistant_text_parts).strip()

    return AgentResult(
        final_text=final_text.strip(),
        steps=steps,
        usage=dict(usage) if isinstance(usage, dict) else usage,
        cost_usd=cost_usd,
        num_turns=num_turns,
        is_error=is_error,
        agent_sdk_name="Claude Agent SDK",
        model_id=model_id,
        raw=result_message,
    )


def _stringify(value: Any) -> str:
    """Return a text representation of a function-tool return value."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)
