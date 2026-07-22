"""Unit tests for the shared agent registry and agent-run declarations."""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

import pytest

from utils.llm import model_runs
from utils.llm.agents import agent_registry, agent_runs
from utils.llm.agents.agent_runs import AgentRun
from utils.llm.agents.results import AgentResult
from utils.llm.agents.tools import (
    FunctionToolSpec,
    MCPStdioServerSpec,
    WebSearchSpec,
    tools_fingerprint,
)

FILENAME_SAFE_NAME_PATTERN = re.compile(r"^k-[a-z0-9~-]+$")
AGENT_RUN_KEY_SUFFIX_PATTERN = re.compile(r"^run-variant-[0-9]{2}$")

# This ledger intentionally duplicates the registry declarations so every new agent run
# requires an explicit historical-key test update, mirroring HISTORICAL_MODEL_RUN_KEYS.
HISTORICAL_AGENT_RUN_KEYS = (
    "claude-haiku-4-5-20251001-claude-agent-sdk-run-variant-01",
    "claude-opus-4-8-claude-agent-sdk-run-variant-01",
    "claude-sonnet-5-claude-agent-sdk-run-variant-01",
    "gpt-5-mini-2025-08-07-openai-agents-sdk-run-variant-01",
    "gpt-5.5-2026-04-23-openai-agents-sdk-run-variant-01",
)


def _example_function_tool(handler) -> FunctionToolSpec:
    """Return a FunctionToolSpec with the given handler and a fixed schema."""
    return FunctionToolSpec(
        name="add",
        description="Add two numbers.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        handler=handler,
    )


# --------------------------------------------------------------------------------------
# Registry invariants
# --------------------------------------------------------------------------------------


def test_historical_ledger_matches_registry() -> None:
    """The explicit historical ledger must match the declared agent-run keys exactly."""
    assert set(HISTORICAL_AGENT_RUN_KEYS) == {run.agent_run_key for run in agent_runs.AGENT_RUNS}


def test_agent_run_keys_follow_convention_and_reference_real_agents() -> None:
    """Every agent run key is <existing_agent_key>-run-variant-XX and is unique."""
    keys = [run.agent_run_key for run in agent_runs.AGENT_RUNS]
    assert len(keys) == len(set(keys)), "agent_run_key values must be unique"

    for run in agent_runs.AGENT_RUNS:
        assert run.agent_key in agent_registry.AGENTS_BY_KEY
        prefix = f"{run.agent_key}-"
        assert run.agent_run_key.startswith(prefix)
        suffix = run.agent_run_key[len(prefix) :]
        assert AGENT_RUN_KEY_SUFFIX_PATTERN.fullmatch(suffix)
        assert FILENAME_SAFE_NAME_PATTERN.fullmatch(run.filename_safe_name)


def test_agent_run_slugs_are_unique() -> None:
    """Slugs are the human-facing lookup key and must be unique across the registry."""
    slugs = [run.slug for run in agent_runs.AGENT_RUNS]
    assert len(slugs) == len(set(slugs))


def test_base_agent_model_provider_matches_sdk() -> None:
    """Each base agent's underlying model routes through the SDK's required provider."""
    for base in agent_registry.AGENTS:
        from utils.llm.provider_registry import PROVIDERS

        assert base.model.provider == PROVIDERS[base.agent_sdk.provider_key]


# --------------------------------------------------------------------------------------
# Validation guardrails
# --------------------------------------------------------------------------------------


def test_agent_run_key_must_match_agent_key() -> None:
    """An agent_run_key that does not extend its agent_key is rejected."""
    base = agent_registry.AGENTS_BY_KEY["claude-opus-4-8-claude-agent-sdk"]
    with pytest.raises(ValueError, match="agent_run_key"):
        AgentRun(
            agent_run_key="not-the-agent-key-run-variant-01",
            slug="bad-key",
            agent=base,
        )


def test_agent_run_key_suffix_must_be_run_variant() -> None:
    """An agent_run_key with a malformed variant suffix is rejected."""
    base = agent_registry.AGENTS_BY_KEY["claude-opus-4-8-claude-agent-sdk"]
    with pytest.raises(ValueError, match="Invalid suffix"):
        AgentRun(
            agent_run_key=f"{base.agent_key}-run-variant-1",  # one digit
            slug="bad-suffix",
            agent=base,
        )


def test_claude_agent_rejects_non_anthropic_model() -> None:
    """A Claude Agent SDK agent must wrap an Anthropic-routed model."""
    with pytest.raises(ValueError, match="requires a Anthropic model"):
        agent_registry.claude_agent(model_key="gpt-5-mini-2025-08-07")


def test_openai_agent_rejects_non_openai_model() -> None:
    """An OpenAI Agents SDK agent must wrap an OpenAI-routed model."""
    with pytest.raises(ValueError, match="requires a OpenAI model"):
        agent_registry.openai_agent(model_key="claude-opus-4-8")


def test_duplicate_agent_run_slug_rejected() -> None:
    """create_agent_runs_list rejects two runs that share a slug."""
    base = agent_registry.AGENTS_BY_KEY["claude-opus-4-8-claude-agent-sdk"]
    run_a = AgentRun(
        agent_run_key=f"{base.agent_key}-run-variant-01",
        slug="dupe-slug",
        agent=base,
        tools=(WebSearchSpec(),),
    )
    run_b = AgentRun(
        agent_run_key=f"{base.agent_key}-run-variant-02",
        slug="dupe-slug",
        agent=base,
        options={"max_turns": 3},
    )
    with pytest.raises(ValueError, match="Duplicate agent-run slug"):
        agent_runs.create_agent_runs_list([run_a, run_b])


def test_duplicate_agent_run_fingerprint_rejected() -> None:
    """Two runs with identical agent/prompt/tools/options collide on fingerprint."""
    base = agent_registry.AGENTS_BY_KEY["claude-opus-4-8-claude-agent-sdk"]
    common = dict(agent=base, system_prompt="x", tools=(WebSearchSpec(),), options={"max_turns": 3})
    run_a = AgentRun(agent_run_key=f"{base.agent_key}-run-variant-01", slug="fp-a", **common)
    run_b = AgentRun(agent_run_key=f"{base.agent_key}-run-variant-02", slug="fp-b", **common)
    with pytest.raises(ValueError, match="Duplicate agent-run fingerprint"):
        agent_runs.create_agent_runs_list([run_a, run_b])


# --------------------------------------------------------------------------------------
# Tool fingerprint contract
# --------------------------------------------------------------------------------------


def test_function_tool_fingerprint_ignores_handler_identity() -> None:
    """Two function tools with the same schema but different handlers fingerprint alike."""
    tool_a = _example_function_tool(lambda a, b: a + b)
    tool_b = _example_function_tool(lambda a, b: a + b + 0)
    assert tools_fingerprint([tool_a]) == tools_fingerprint([tool_b])


def test_tool_fingerprint_distinguishes_configuration() -> None:
    """Different tool configurations produce different fingerprints."""
    assert tools_fingerprint([WebSearchSpec()]) != tools_fingerprint([WebSearchSpec(max_uses=3)])
    stdio = MCPStdioServerSpec(name="fs", command="npx", args=("server",))
    assert tools_fingerprint([stdio]) != tools_fingerprint([WebSearchSpec()])


# --------------------------------------------------------------------------------------
# Drop-in ModelRun compatibility contract
# --------------------------------------------------------------------------------------


@runtime_checkable
class _BenchmarkRun(Protocol):
    """The surface ForecastBench drives on a run; both ModelRun and AgentRun must satisfy it."""

    model_run_key: str
    slug: str
    provider_model_id: str

    def get_response(self, prompt: str) -> str: ...


def test_agent_run_satisfies_benchmark_run_surface() -> None:
    """An AgentRun exposes the same surface ForecastBench uses on a ModelRun."""
    agent_run = agent_runs.get_agent_run("claude-opus-4-8-claude-agent-sdk-run-variant-01")
    model_run = model_runs.get_model_run("claude-opus-4-8-run-variant-01")

    # Independent source of truth: whatever the model run exposes for these members,
    # the agent run must expose too, with the run key mirrored onto model_run_key.
    assert isinstance(model_run, _BenchmarkRun)
    assert isinstance(agent_run, _BenchmarkRun)

    for member in (
        "provider",
        "lab",
        "release_date",
        "options",
        "display_name",
        "filename_safe_name",
    ):
        assert hasattr(agent_run, member), member

    assert agent_run.model_run_key == agent_run.agent_run_key
    assert agent_run.provider == agent_run.agent.provider
    assert agent_run.lab == agent_run.agent.lab


def test_get_response_returns_final_text_and_merges_option_overrides(monkeypatch) -> None:
    """get_response returns the run's final text and forwards merged option overrides."""
    captured: dict = {}

    def fake_run_agent(*, base_agent, system_prompt, prompt, tools, options) -> AgentResult:
        captured.update(
            base_agent=base_agent,
            system_prompt=system_prompt,
            prompt=prompt,
            tools=tools,
            options=options,
        )
        return AgentResult(final_text="the-answer", steps=[])

    monkeypatch.setattr(agent_registry, "run_agent", fake_run_agent)

    base = agent_registry.AGENTS_BY_KEY["gpt-5-mini-2025-08-07-openai-agents-sdk"]
    run = AgentRun(
        agent_run_key=f"{base.agent_key}-run-variant-01",
        slug="merge-test",
        agent=base,
        system_prompt="sys",
        tools=(WebSearchSpec(),),
        options={"max_turns": 5, "temperature": 0},
    )

    answer = run.get_response("hello", temperature=1)

    assert answer == "the-answer"
    assert captured["prompt"] == "hello"
    assert captured["system_prompt"] == "sys"
    # Base options are preserved and per-call overrides win.
    assert captured["options"] == {"max_turns": 5, "temperature": 1}
    # The base run's declared options are not mutated by the override.
    assert run.options == {"max_turns": 5, "temperature": 0}


# --------------------------------------------------------------------------------------
# Tool translation (SDK-dependent; skipped without the optional [agents] extra)
# --------------------------------------------------------------------------------------


def test_claude_tool_translation_maps_categories() -> None:
    """The Claude provider maps web search, function, and MCP specs to CLI options."""
    pytest.importorskip("claude_agent_sdk")
    from utils.llm.agents.providers.anthropic_agent import ClaudeAgentSDKProvider

    provider = ClaudeAgentSDKProvider(api_key="test-key")
    allowed, mcp_servers, enabled_builtins = provider._translate_tools(
        [
            WebSearchSpec(),
            _example_function_tool(lambda a, b: a + b),
            MCPStdioServerSpec(name="fs", command="npx", args=("server",)),
        ]
    )

    assert "WebSearch" in allowed
    assert "WebSearch" in enabled_builtins
    assert "mcp__fri_functions__add" in allowed
    assert "fri_functions" in mcp_servers
    assert "mcp__fs" in allowed
    assert mcp_servers["fs"]["command"] == "npx"


def test_claude_default_denylist_blocks_filesystem_tools() -> None:
    """With no web tools declared, built-in filesystem/exec tools are disallowed."""
    pytest.importorskip("claude_agent_sdk")
    from utils.llm.agents.providers.anthropic_agent import (
        _DEFAULT_DISALLOWED_TOOLS,
        ClaudeAgentSDKProvider,
    )

    provider = ClaudeAgentSDKProvider(api_key="test-key")
    allowed, _servers, enabled = provider._translate_tools([WebSearchSpec()])

    # WebSearch is now opt-in, so it should not remain in the effective denylist.
    effective_denylist = [name for name in _DEFAULT_DISALLOWED_TOOLS if name not in enabled]
    assert "Bash" in effective_denylist
    assert "Read" in effective_denylist
    assert "WebSearch" not in effective_denylist
    assert "WebSearch" in allowed
