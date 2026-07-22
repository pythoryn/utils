"""Base agent provider with a robust async-to-sync bridge and shared retry.

External agent SDKs are asynchronous, but the shared registry exposes a synchronous
``get_response``/``run`` so agent runs are drop-in replacements for model runs (including
inside ForecastBench's thread-pool runner). :func:`run_coro_blocking` bridges the two by
always executing the coroutine in a dedicated thread with a fresh event loop, which is safe
whether the caller is a plain script, a ForecastBench worker thread, or a Jupyter cell that
already has a running loop.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine, Sequence
from typing import Any, Final

from ...utils import get_response_with_retry
from ..results import AgentResult
from ..tools import ToolSpec

_DEFAULT_WAIT_TIME_SECONDS: Final[int] = 30


def _new_event_loop() -> asyncio.AbstractEventLoop:
    """Return a fresh event loop that supports subprocesses in any host context.

    On Windows, agent SDKs that spawn a CLI subprocess (e.g. the Claude Agent SDK) require a
    Proactor loop. A host such as Jupyter/IPython may have installed the Selector loop policy,
    which cannot spawn subprocesses on Windows, so build a Proactor loop explicitly rather than
    inheriting the ambient policy.
    """
    if sys.platform == "win32":
        return asyncio.ProactorEventLoop()
    return asyncio.new_event_loop()


def run_coro_blocking(coro_factory: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
    """Run a coroutine to completion from synchronous code, in any loop context.

    ``coro_factory`` is called inside the worker thread so the coroutine and any SDK
    clients it creates are bound to that thread's fresh event loop.
    """
    box: dict[str, Any] = {}

    def _worker() -> None:
        loop = _new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            box["value"] = loop.run_until_complete(coro_factory())
        except BaseException as exc:  # noqa: BLE001,B036 - captured and re-raised below
            box["error"] = exc
        finally:
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)

    thread = threading.Thread(target=_worker, name="agent-run-loop")
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


class BaseAgentProvider(ABC):
    """Abstract agent provider that wraps an SDK's async agent loop with retry logic."""

    retry_message: str = "Agent run failed."

    def __init__(self, *, api_key: str, default_wait_time: int | None = None) -> None:
        """Store the provider API key and optional retry backoff interval."""
        if not api_key:
            raise ValueError(
                f"API key required for {type(self).__name__}. "
                "Call configure_api_keys() (or configure_api_keys(from_gcp=True))."
            )
        self._api_key = api_key
        self._default_wait_time = default_wait_time or _DEFAULT_WAIT_TIME_SECONDS

    def run(
        self,
        *,
        model_id: str,
        system_prompt: str | None,
        prompt: str,
        tools: Sequence[ToolSpec],
        options: dict[str, Any],
    ) -> AgentResult:
        """Run the agent for one prompt, retrying on provider errors."""

        def api_call() -> AgentResult:
            return run_coro_blocking(
                lambda: self._arun(
                    model_id=model_id,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    tools=tools,
                    options=options,
                )
            )

        return get_response_with_retry(
            api_call,
            self._default_wait_time,
            self.retry_message,
        )

    @abstractmethod
    async def _arun(
        self,
        *,
        model_id: str,
        system_prompt: str | None,
        prompt: str,
        tools: Sequence[ToolSpec],
        options: dict[str, Any],
    ) -> AgentResult:
        """Execute one agent run against the underlying SDK and return a normalized result."""
