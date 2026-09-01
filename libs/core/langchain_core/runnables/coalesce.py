"""Request coalescing wrapper for ``Runnable`` objects."""

from __future__ import annotations

import asyncio
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict
from typing_extensions import override

from langchain_core.load.dump import dumpd
from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    get_config_list,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from langchain_core.runnables.graph import Graph
    from langchain_core.runnables.schema import StreamEvent


_CANCELLED = object()


def _make_coalesce_key(input_: Any) -> str:
    """Create a deterministic key from an input value.

    Dict ordering is ignored via ``sort_keys``.
    """
    return json.dumps(dumpd(input_), sort_keys=True)


@dataclass
class CoalesceStats:
    """Statistics for a coalesce backend."""

    active: int = 0
    coalesced: int = 0
    total: int = 0


class CoalesceBackend(ABC):
    """Abstract base class for coalesce backends."""

    @abstractmethod
    def register(self, key: str) -> bool:
        """Register a key for coalescing.

        Returns:
            ``True`` if this caller is the leader (first to register),
            ``False`` if this caller is a joiner (key already active).
        """

    @abstractmethod
    def join(self, key: str) -> Any:
        """Block until the leader completes and return the result.

        Raises:
            Exception: Re-raises the leader's exception if one occurred.
            asyncio.CancelledError: If the flight was cancelled via
                ``coalesce_clear``.
        """

    @abstractmethod
    def complete(
        self,
        key: str,
        *,
        result: Any | None = None,
        error: Exception | None = None,
    ) -> None:
        """Mark a key as complete, storing its result or error.

        All joiners waiting on this key will be unblocked.
        """

    @abstractmethod
    def is_active(self, key: str) -> bool:
        """Return whether a key currently has an active flight."""

    @property
    @abstractmethod
    def stats(self) -> CoalesceStats:
        """Current coalescing statistics."""

    @abstractmethod
    def clear(self) -> None:
        """Cancel all in-flight keys and reset statistics.

        Waiters receive ``asyncio.CancelledError``.
        """

    async def aregister(self, key: str) -> bool:
        """Async version of ``register``."""
        return self.register(key)

    async def ajoin(self, key: str) -> Any:
        """Async version of ``join``."""
        return self.join(key)

    async def acomplete(
        self,
        key: str,
        *,
        result: Any | None = None,
        error: Exception | None = None,
    ) -> None:
        """Async version of ``complete``."""
        self.complete(key, result=result, error=error)

    async def ais_active(self, key: str) -> bool:
        """Async version of ``is_active``."""
        return self.is_active(key)

    @property
    async def astats(self) -> CoalesceStats:
        """Async version of ``stats``."""
        return self.stats

    async def aclear(self) -> None:
        """Async version of ``clear``."""
        self.clear()


class _Flight:
    """Internal bookkeeping shared by sync and async callers."""

    __slots__ = ("error", "event", "result", "sync_event")

    def __init__(self) -> None:
        self.event = asyncio.Event()
        self.sync_event = threading.Event()
        self.result: Any = None
        self.error: Exception | None = None


class InMemoryCoalesceBackend(CoalesceBackend):
    """Thread-safe in-memory coalesce backend."""

    def __init__(self) -> None:
        """Initialize an empty in-memory backend."""
        self._lock = threading.Lock()
        self._flights: dict[str, _Flight] = {}
        self._total = 0
        self._coalesced = 0

    @override
    def register(self, key: str) -> bool:
        with self._lock:
            self._total += 1
            if key in self._flights:
                self._coalesced += 1
                return False
            self._flights[key] = _Flight()
            return True

    @override
    def join(self, key: str) -> Any:
        with self._lock:
            flight = self._flights.get(key)
        if flight is None:
            msg = f"No active flight for key: {key!r}"
            raise KeyError(msg)
        flight.sync_event.wait()
        if flight.result is _CANCELLED:
            raise asyncio.CancelledError
        if flight.error is not None:
            raise flight.error
        return flight.result

    @override
    def complete(
        self,
        key: str,
        *,
        result: Any | None = None,
        error: Exception | None = None,
    ) -> None:
        with self._lock:
            flight = self._flights.get(key)
            if flight is None:
                return
            flight.result = result
            flight.error = error
            self._flights.pop(key, None)
        flight.sync_event.set()
        flight.event.set()

    @override
    def is_active(self, key: str) -> bool:
        with self._lock:
            return key in self._flights

    @property
    @override
    def stats(self) -> CoalesceStats:
        with self._lock:
            active = len(self._flights)
            return CoalesceStats(
                active=active,
                coalesced=self._coalesced,
                total=self._total,
            )

    @override
    def clear(self) -> None:
        with self._lock:
            flights = dict(self._flights)
            self._flights.clear()
            self._total = 0
            self._coalesced = 0
        for flight in flights.values():
            flight.result = _CANCELLED
            flight.sync_event.set()
            flight.event.set()

    @override
    async def aregister(self, key: str) -> bool:
        with self._lock:
            self._total += 1
            if key in self._flights:
                self._coalesced += 1
                return False
            self._flights[key] = _Flight()
            return True

    @override
    async def ajoin(self, key: str) -> Any:
        with self._lock:
            flight = self._flights.get(key)
        if flight is None:
            msg = f"No active flight for key: {key!r}"
            raise KeyError(msg)
        await flight.event.wait()
        if flight.result is _CANCELLED:
            raise asyncio.CancelledError
        if flight.error is not None:
            raise flight.error
        return flight.result

    @override
    async def acomplete(
        self,
        key: str,
        *,
        result: Any | None = None,
        error: Exception | None = None,
    ) -> None:
        with self._lock:
            flight = self._flights.get(key)
            if flight is None:
                return
            flight.result = result
            flight.error = error
            self._flights.pop(key, None)
        flight.sync_event.set()
        flight.event.set()


class RunnableCoalesce(RunnableSerializable[Input, Output]):
    """Wrap a ``Runnable`` with request coalescing logic.

    When multiple concurrent callers invoke the runnable with the same input,
    only one execution runs and all callers receive the result.
    """

    bound: Runnable[Input, Output]
    backend: CoalesceBackend

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        bound: Runnable[Input, Output],
        *,
        backend: CoalesceBackend,
        **kwargs: Any,
    ) -> None:
        """Wrap a runnable with the supplied coalescing backend."""
        super().__init__(bound=bound, backend=backend, **kwargs)

    @property
    @override
    def InputType(self) -> type[Input]:
        return self.bound.InputType

    @property
    @override
    def OutputType(self) -> type[Output]:
        return self.bound.OutputType

    @override
    def get_input_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        return self.bound.get_input_schema(config)

    @override
    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        return self.bound.get_output_schema(config)

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return self.bound.config_specs

    @override
    def get_graph(self, config: RunnableConfig | None = None) -> Graph:
        return self.bound.get_graph(config)

    @override
    def get_name(
        self, suffix: str | None = None, *, name: str | None = None
    ) -> str:
        name = name or self.name or f"Coalesce<{self.bound.get_name()}>"
        return super().get_name(suffix, name=name)

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return False

    def coalesce_info(self) -> CoalesceStats:
        """Return current coalescing statistics."""
        return self.backend.stats

    def coalesce_clear(self) -> None:
        """Cancel all in-flight coalesced calls and reset statistics.

        Waiters receive ``asyncio.CancelledError``.
        """
        self.backend.clear()

    def _fire_joined_callbacks(
        self,
        config: RunnableConfig,
        input_: Input,
        output: Output,
    ) -> None:
        """Fire chain-start/chain-end callbacks for a joined caller."""
        config = ensure_config(config)
        cb_manager = get_callback_manager_for_config(config)
        run_manager = cb_manager.on_chain_start(
            None,
            dumpd(input_),
            name=self.get_name(),
        )
        run_manager.on_chain_end(dumpd(output))

    def _fire_joined_error_callbacks(
        self,
        config: RunnableConfig,
        input_: Input,
        error: Exception,
    ) -> None:
        """Fire chain-start/chain-error callbacks for a joined caller."""
        config = ensure_config(config)
        cb_manager = get_callback_manager_for_config(config)
        run_manager = cb_manager.on_chain_start(
            None,
            dumpd(input_),
            name=self.get_name(),
        )
        run_manager.on_chain_error(error)

    async def _afire_joined_callbacks(
        self,
        config: RunnableConfig,
        input_: Input,
        output: Output,
    ) -> None:
        """Async fire chain-start/chain-end callbacks for a joined caller."""
        config = ensure_config(config)
        cb_manager = get_async_callback_manager_for_config(config)
        run_manager = await cb_manager.on_chain_start(
            None,
            dumpd(input_),
            name=self.get_name(),
        )
        await run_manager.on_chain_end(dumpd(output))

    async def _afire_joined_error_callbacks(
        self,
        config: RunnableConfig,
        input_: Input,
        error: Exception,
    ) -> None:
        """Async fire chain-start/chain-error callbacks for a joined caller."""
        config = ensure_config(config)
        cb_manager = get_async_callback_manager_for_config(config)
        run_manager = await cb_manager.on_chain_start(
            None,
            dumpd(input_),
            name=self.get_name(),
        )
        await run_manager.on_chain_error(error)

    @override
    def invoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        config = ensure_config(config)
        key = _make_coalesce_key(input)
        is_leader = self.backend.register(key)
        if is_leader:
            try:
                result = self.bound.invoke(input, config, **kwargs)
            except Exception as exc:
                self.backend.complete(key, error=exc)
                raise
            self.backend.complete(key, result=result)
            return result
        try:
            result = self.backend.join(key)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fire_joined_error_callbacks(config, input, exc)
            raise
        self._fire_joined_callbacks(config, input, result)
        return cast("Output", result)

    @override
    async def ainvoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        config = ensure_config(config)
        key = _make_coalesce_key(input)
        is_leader = await self.backend.aregister(key)
        if is_leader:
            try:
                result = await self.bound.ainvoke(input, config, **kwargs)
            except Exception as exc:
                await self.backend.acomplete(key, error=exc)
                raise
            await self.backend.acomplete(key, result=result)
            return result
        try:
            result = await self.backend.ajoin(key)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._afire_joined_error_callbacks(config, input, exc)
            raise
        await self._afire_joined_callbacks(config, input, result)
        return cast("Output", result)

    @override
    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        config = ensure_config(config)
        key = _make_coalesce_key(input)
        is_leader = self.backend.register(key)
        if is_leader:
            chunks: list[Output] = []
            try:
                for chunk in self.bound.stream(input, config, **kwargs):
                    chunks.append(chunk)
                    yield chunk
            except Exception as exc:
                self.backend.complete(key, error=exc)
                raise
            self.backend.complete(key, result=chunks)
        else:
            try:
                result = self.backend.join(key)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._fire_joined_error_callbacks(config, input, exc)
                raise
            self._fire_joined_callbacks(
                config, input, result[-1] if result else cast("Output", None)
            )
            yield from result

    @override
    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        config = ensure_config(config)
        key = _make_coalesce_key(input)
        is_leader = await self.backend.aregister(key)
        if is_leader:
            chunks: list[Output] = []
            try:
                async for chunk in self.bound.astream(input, config, **kwargs):
                    chunks.append(chunk)
                    yield chunk
            except Exception as exc:
                await self.backend.acomplete(key, error=exc)
                raise
            await self.backend.acomplete(key, result=chunks)
        else:
            try:
                result = await self.backend.ajoin(key)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._afire_joined_error_callbacks(config, input, exc)
                raise
            await self._afire_joined_callbacks(
                config,
                input,
                result[-1] if result else cast("Output", None),
            )
            for chunk in result:
                yield chunk

    @override
    def transform(
        self,
        input: Iterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        yield from self.bound.transform(input, config, **kwargs)

    @override
    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        async for chunk in self.bound.atransform(input, config, **kwargs):
            yield chunk

    @override
    def batch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Output]:
        if not inputs:
            return []
        configs = get_config_list(config, len(inputs))

        keys = [_make_coalesce_key(inp) for inp in inputs]
        key_to_first: dict[str, int] = {}
        unique_indices: list[int] = []
        duplicate_map: dict[int, int] = {}

        for i, k in enumerate(keys):
            if k not in key_to_first:
                key_to_first[k] = i
                unique_indices.append(i)
            else:
                duplicate_map[i] = key_to_first[k]

        unique_inputs = [inputs[i] for i in unique_indices]
        unique_configs = [configs[i] for i in unique_indices]
        unique_results = super().batch(
            unique_inputs,
            unique_configs,
            return_exceptions=return_exceptions,
            **kwargs,
        )

        result_by_idx: dict[int, Output] = {}
        for local_idx, orig_idx in enumerate(unique_indices):
            result_by_idx[orig_idx] = unique_results[local_idx]

        ordered: list[Output] = []
        for i in range(len(inputs)):
            if i in duplicate_map:
                leader_idx = duplicate_map[i]
                res = result_by_idx[leader_idx]
                if isinstance(res, Exception) and not return_exceptions:
                    raise res
                if isinstance(res, Exception):
                    self._fire_joined_error_callbacks(configs[i], inputs[i], res)
                else:
                    self._fire_joined_callbacks(configs[i], inputs[i], res)
                ordered.append(res)
            else:
                ordered.append(result_by_idx[i])
        return ordered

    @override
    async def abatch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Output]:
        if not inputs:
            return []
        configs = get_config_list(config, len(inputs))

        keys = [_make_coalesce_key(inp) for inp in inputs]
        key_to_first: dict[str, int] = {}
        unique_indices: list[int] = []
        duplicate_map: dict[int, int] = {}

        for i, k in enumerate(keys):
            if k not in key_to_first:
                key_to_first[k] = i
                unique_indices.append(i)
            else:
                duplicate_map[i] = key_to_first[k]

        unique_inputs = [inputs[i] for i in unique_indices]
        unique_configs = [configs[i] for i in unique_indices]
        unique_results = await super().abatch(
            unique_inputs,
            unique_configs,
            return_exceptions=return_exceptions,
            **kwargs,
        )

        result_by_idx: dict[int, Output] = {}
        for local_idx, orig_idx in enumerate(unique_indices):
            result_by_idx[orig_idx] = unique_results[local_idx]

        ordered: list[Output] = []
        for i in range(len(inputs)):
            if i in duplicate_map:
                leader_idx = duplicate_map[i]
                res = result_by_idx[leader_idx]
                if isinstance(res, Exception) and not return_exceptions:
                    raise res
                if isinstance(res, Exception):
                    await self._afire_joined_error_callbacks(configs[i], inputs[i], res)
                else:
                    await self._afire_joined_callbacks(configs[i], inputs[i], res)
                ordered.append(res)
            else:
                ordered.append(result_by_idx[i])
        return ordered

    @override
    def batch_as_completed(
        self,
        inputs: Sequence[Input],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> Iterator[tuple[int, Output | Exception]]:
        if not inputs:
            return
        configs = get_config_list(config, len(inputs))

        keys = [_make_coalesce_key(inp) for inp in inputs]
        key_to_first: dict[str, int] = {}
        unique_indices: list[int] = []
        duplicates_of: dict[int, list[int]] = {}

        for i, k in enumerate(keys):
            if k not in key_to_first:
                key_to_first[k] = i
                unique_indices.append(i)
                duplicates_of[i] = []
            else:
                duplicates_of[key_to_first[k]].append(i)

        unique_inputs = [inputs[i] for i in unique_indices]
        unique_configs = [configs[i] for i in unique_indices]

        for local_idx, result in super().batch_as_completed(
            unique_inputs,
            unique_configs,
            return_exceptions=return_exceptions,
            **kwargs,
        ):
            orig_idx = unique_indices[local_idx]
            yield (orig_idx, result)
            for dup_idx in duplicates_of[orig_idx]:
                if isinstance(result, Exception):
                    if not return_exceptions:
                        raise result
                    self._fire_joined_error_callbacks(
                        configs[dup_idx], inputs[dup_idx], result
                    )
                else:
                    self._fire_joined_callbacks(
                        configs[dup_idx], inputs[dup_idx], result
                    )
                yield (dup_idx, result)

    @override
    async def abatch_as_completed(
        self,
        inputs: Sequence[Input],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[int, Output | Exception]]:
        if not inputs:
            return
        configs = get_config_list(config, len(inputs))

        keys = [_make_coalesce_key(inp) for inp in inputs]
        key_to_first: dict[str, int] = {}
        unique_indices: list[int] = []
        duplicates_of: dict[int, list[int]] = {}

        for i, k in enumerate(keys):
            if k not in key_to_first:
                key_to_first[k] = i
                unique_indices.append(i)
                duplicates_of[i] = []
            else:
                duplicates_of[key_to_first[k]].append(i)

        unique_inputs = [inputs[i] for i in unique_indices]
        unique_configs = [configs[i] for i in unique_indices]

        async for local_idx, result in super().abatch_as_completed(
            unique_inputs,
            unique_configs,
            return_exceptions=return_exceptions,
            **kwargs,
        ):
            orig_idx = unique_indices[local_idx]
            yield (orig_idx, result)
            for dup_idx in duplicates_of[orig_idx]:
                if isinstance(result, Exception):
                    if not return_exceptions:
                        raise result
                    await self._afire_joined_error_callbacks(
                        configs[dup_idx], inputs[dup_idx], result
                    )
                else:
                    await self._afire_joined_callbacks(
                        configs[dup_idx], inputs[dup_idx], result
                    )
                yield (dup_idx, result)

    @override
    async def astream_events(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2"] = "v2",
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        async for event in self.bound.astream_events(
            input,
            config,
            version=version,
            include_names=include_names,
            include_types=include_types,
            include_tags=include_tags,
            exclude_names=exclude_names,
            exclude_types=exclude_types,
            exclude_tags=exclude_tags,
            **kwargs,
        ):
            yield event
