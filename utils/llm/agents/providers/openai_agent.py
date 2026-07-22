"""Agent provider backed by the OpenAI Agents SDK.

Translates provider-neutral :mod:`~utils.llm.agents.tools` specs into an ``agents.Agent``
plus tool objects and drives one prompt through ``agents.Runner.run``. The underlying model
is bound to an explicit ``AsyncOpenAI`` client so the run uses the API key configured through
the shared registry rather than ambient environment state. Tracing is disabled per run.
"""

from __future__ import annotations

import contextlib
import inspect
import json
from collections.abc import Sequence
from typing import Any, Final

from ..results import (
    STEP_OTHER,
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

# ModelSettings fields passed straight through from an agent run's ``options``.
_MODEL_SETTINGS_KEYS: Final[frozenset[str]] = frozenset(
    {
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "tool_choice",
        "parallel_tool_calls",
        "truncation",
        "max_tokens",
        "reasoning",
        "verbosity",
        "extra_body",
        "extra_args",
    }
)
_DEFAULT_MAX_TURNS: Final[int] = 20


class OpenAIAgentsProvider(BaseAgentProvider):
    """Agent provider that runs agents through the OpenAI Agents SDK."""

    retry_message = "OpenAI Agents SDK run failed."

    async def _arun(
        self,
        *,
        model_id: str,
        system_prompt: str | None,
        prompt: str,
        tools: Sequence[ToolSpec],
        options: dict[str, Any],
    ) -> AgentResult:
        from agents import (
            Agent,
            ModelSettings,
            OpenAIResponsesModel,
            RunConfig,
            Runner,
        )
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self._api_key)
        model = OpenAIResponsesModel(model=model_id, openai_client=client)

        model_settings = ModelSettings(
            **{key: options[key] for key in _MODEL_SETTINGS_KEYS if key in options}
        )

        async with contextlib.AsyncExitStack() as stack:
            # Close the client's httpx connection pool while this run's event loop is still
            # alive. run_coro_blocking() closes that loop as soon as the run returns, so an
            # unclosed pool would be torn down later by the GC on an already-closed loop,
            # raising "Event loop is closed" from a never-awaited task (noisy on the Windows
            # Proactor loop in particular).
            stack.push_async_callback(client.close)
            tool_objects, mcp_servers = await self._translate_tools(tools, stack)
            agent = Agent(
                name=options.get("agent_name", "fri-agent"),
                instructions=system_prompt,
                model=model,
                model_settings=model_settings,
                tools=tool_objects,
                mcp_servers=mcp_servers,
            )
            run_result = await Runner.run(
                agent,
                prompt,
                max_turns=int(options.get("max_turns", _DEFAULT_MAX_TURNS)),
                run_config=RunConfig(tracing_disabled=True),
            )

        return _build_result(model_id=model_id, run_result=run_result)

    async def _translate_tools(
        self, tools: Sequence[ToolSpec], stack: contextlib.AsyncExitStack
    ) -> tuple[list[Any], list[Any]]:
        """Return (tool_objects, mcp_server_objects) for the OpenAI Agents SDK."""
        tool_objects: list[Any] = []
        mcp_servers: list[Any] = []

        for tool in tools:
            if isinstance(tool, WebSearchSpec):
                tool_objects.append(_web_search_tool(tool))
            elif isinstance(tool, WebFetchSpec):
                # The OpenAI Agents SDK has no standalone web-fetch tool; skip it.
                continue
            elif isinstance(tool, FunctionToolSpec):
                tool_objects.append(_function_tool(tool))
            elif isinstance(tool, MCPHttpServerSpec):
                tool_objects.append(_hosted_mcp_tool(tool))
            elif isinstance(tool, MCPStdioServerSpec):
                mcp_servers.append(await _stdio_mcp_server(tool, stack))
            else:
                raise TypeError(f"Unsupported tool spec for OpenAI Agents SDK: {tool!r}")

        return tool_objects, mcp_servers


def _web_search_tool(spec: WebSearchSpec) -> Any:
    """Build an OpenAI ``WebSearchTool`` from a WebSearchSpec."""
    from agents import WebSearchTool

    kwargs: dict[str, Any] = {"search_context_size": spec.search_context_size}
    if spec.allowed_domains:
        with contextlib.suppress(Exception):
            from agents import WebSearchToolFilters

            kwargs["filters"] = WebSearchToolFilters(allowed_domains=list(spec.allowed_domains))
    return WebSearchTool(**kwargs)


def _function_tool(spec: FunctionToolSpec) -> Any:
    """Build an OpenAI ``FunctionTool`` from a FunctionToolSpec."""
    from agents import FunctionTool

    async def _on_invoke(_ctx: Any, args_json: str) -> str:
        arguments = json.loads(args_json) if args_json else {}
        outcome = spec.handler(**arguments)
        if inspect.isawaitable(outcome):
            outcome = await outcome
        return _stringify(outcome)

    return FunctionTool(
        name=spec.name,
        description=spec.description,
        params_json_schema=dict(spec.parameters),
        on_invoke_tool=_on_invoke,
        strict_json_schema=False,
    )


def _hosted_mcp_tool(spec: MCPHttpServerSpec) -> Any:
    """Build an OpenAI ``HostedMCPTool`` for a remote MCP server."""
    from agents import HostedMCPTool

    tool_config: dict[str, Any] = {
        "type": "mcp",
        "server_label": spec.name,
        "server_url": spec.url,
        "require_approval": spec.require_approval,
    }
    if spec.headers:
        tool_config["headers"] = dict(spec.headers)
    if spec.tool_names:
        tool_config["allowed_tools"] = list(spec.tool_names)
    return HostedMCPTool(tool_config=tool_config)


async def _stdio_mcp_server(spec: MCPStdioServerSpec, stack: contextlib.AsyncExitStack) -> Any:
    """Connect a local stdio MCP server and register it for cleanup on the exit stack."""
    from agents.mcp import MCPServerStdio

    params: dict[str, Any] = {"command": spec.command, "args": list(spec.args)}
    if spec.env:
        params["env"] = dict(spec.env)
    server = MCPServerStdio(params=params, name=spec.name, cache_tools_list=True)
    return await stack.enter_async_context(server)


def _build_result(*, model_id: str, run_result: Any) -> AgentResult:
    """Assemble a normalized AgentResult from an OpenAI Agents ``RunResult``."""
    final_output = getattr(run_result, "final_output", None)
    final_text = final_output if isinstance(final_output, str) else _stringify(final_output)

    steps = [
        step
        for item in getattr(run_result, "new_items", []) or []
        if (step := _run_item_to_step(item)) is not None
    ]
    usage = _aggregate_usage(getattr(run_result, "raw_responses", []) or [])
    num_turns = len(getattr(run_result, "raw_responses", []) or []) or None

    return AgentResult(
        final_text=(final_text or "").strip(),
        steps=steps,
        usage=usage,
        cost_usd=None,
        num_turns=num_turns,
        is_error=False,
        agent_sdk_name="OpenAI Agents SDK",
        model_id=model_id,
        raw=run_result,
    )


def _run_item_to_step(item: Any) -> AgentStep | None:
    """Normalize one OpenAI Agents ``RunItem`` into an AgentStep."""
    item_type = type(item).__name__
    raw = getattr(item, "raw_item", None)

    if item_type == "MessageOutputItem":
        return AgentStep(kind=STEP_TEXT, text=_message_text(item), raw_type=item_type)
    if item_type == "ReasoningItem":
        return AgentStep(kind=STEP_REASONING, text=_reasoning_text(raw), raw_type=item_type)
    if item_type in ("ToolCallItem", "MCPListToolsItem", "HandoffCallItem"):
        return AgentStep(
            kind=STEP_TOOL_USE,
            tool_name=getattr(raw, "name", None) or getattr(raw, "type", None),
            tool_input=_tool_call_arguments(raw),
            raw_type=item_type,
        )
    if item_type in ("ToolCallOutputItem", "MCPApprovalResponseItem"):
        return AgentStep(
            kind=STEP_TOOL_RESULT,
            tool_output=getattr(item, "output", None),
            raw_type=item_type,
        )
    return AgentStep(kind=STEP_OTHER, raw_type=item_type)


def _message_text(item: Any) -> str | None:
    """Return the assistant text carried by a MessageOutputItem."""
    with contextlib.suppress(Exception):
        from agents import ItemHelpers

        return ItemHelpers.text_message_output(item)
    return None


def _reasoning_text(raw: Any) -> str | None:
    """Return summarized reasoning text from a reasoning item's raw payload."""
    summary = getattr(raw, "summary", None)
    if not summary:
        return None
    parts = [getattr(entry, "text", None) for entry in summary]
    return "\n".join(part for part in parts if part) or None


def _tool_call_arguments(raw: Any) -> Any:
    """Return the arguments of a tool call, parsed from JSON when possible."""
    arguments = getattr(raw, "arguments", None)
    if isinstance(arguments, str):
        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(arguments)
    return arguments


def _aggregate_usage(raw_responses: Sequence[Any]) -> dict[str, Any] | None:
    """Sum token usage across the model responses in a run."""
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    found = False
    for response in raw_responses:
        usage = getattr(response, "usage", None)
        if usage is None:
            continue
        found = True
        input_tokens += getattr(usage, "input_tokens", 0) or 0
        output_tokens += getattr(usage, "output_tokens", 0) or 0
        total_tokens += getattr(usage, "total_tokens", 0) or 0
    if not found:
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _stringify(value: Any) -> str:
    """Return a text representation of a tool return value or final output."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)
