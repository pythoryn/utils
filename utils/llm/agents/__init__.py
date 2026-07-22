"""Run agents through external agent SDKs behind a model-run-compatible interface.

This subpackage mirrors the model registry/model-run layering for agents:

* :mod:`agent_sdk_registry` enumerates external agent frameworks (Claude Agent SDK, OpenAI
  Agents SDK), analogous to the model provider registry.
* :mod:`agent_registry` defines base :class:`~utils.llm.agents.agent_registry.Agent` entries
  (an agent SDK bound to a base model), analogous to ``Model``.
* :mod:`agent_runs` defines :class:`~utils.llm.agents.agent_runs.AgentRun` (a base agent plus
  system prompt, tools, and options), analogous to ``ModelRun`` and drop-in compatible with it.
* :mod:`tools` declares provider-neutral tool specs; :mod:`providers` translates them per SDK.
* :mod:`transcript` captures agent trajectories to Markdown/JSONL.

Importing this package does not import the external SDKs; those are imported lazily only when an
agent actually runs, so ``import utils.llm.agents`` works without the optional ``[agents]`` extra.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_runs import (
        ACTIVE_AGENT_RUNS,
        AGENT_RUNS,
        AgentRun,
        get_agent_run,
        get_agent_run_by_slug,
        select_agent_runs,
    )

_AGENT_RUN_EXPORTS = {
    "ACTIVE_AGENT_RUNS",
    "ACTIVE_AGENT_RUNS_BY_KEY",
    "ACTIVE_AGENT_RUNS_BY_SLUG",
    "AGENT_RUNS",
    "AGENT_RUNS_BY_KEY",
    "AGENT_RUNS_BY_SLUG",
    "AgentRun",
    "get_agent_run",
    "get_agent_run_by_slug",
    "select_agent_runs",
}

_SUBMODULES = {
    "agent_registry",
    "agent_runs",
    "agent_sdk_registry",
    "providers",
    "results",
    "tools",
    "transcript",
}

__all__ = [
    "ACTIVE_AGENT_RUNS",
    "AGENT_RUNS",
    "AgentRun",
    "agent_registry",
    "agent_runs",
    "agent_sdk_registry",
    "get_agent_run",
    "get_agent_run_by_slug",
    "results",
    "select_agent_runs",
    "tools",
    "transcript",
]


def __getattr__(name: str):
    """Lazily import agent submodules and run exports on attribute access."""
    if name in _AGENT_RUN_EXPORTS:
        module = import_module(f"{__name__}.agent_runs")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
