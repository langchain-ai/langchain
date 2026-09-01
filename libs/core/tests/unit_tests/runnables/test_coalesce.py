from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
)
from langchain_core.runnables.coalesce import (
    CoalesceBackend,
    CoalesceStats,
    InMemoryCoalesceBackend,
)
from tests.unit_tests.fake.callbacks import FakeCallbackHandler
class _Blocking(Runnable[str, str]):
    """Blocks until released - useful for proving concurrent coalescing."""

    def __init__(self) -> None:
        self.call_count = 0
        self._event = threading.Event()
        self._lock = threading.Lock()

    @property
    def InputType(self) -> type[str]:
        return str

    @property
    def OutputType(self) -> type[str]:
        return str

    def invoke(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> str:
        with self._lock:
            self.call_count += 1
        self._event.wait(timeout=10)
        return input.upper()

    async def ainvoke(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> str:
        with self._lock:
            self.call_count += 1
        while not self._event.is_set():
            await asyncio.sleep(0.01)
        return input.upper()

    def release(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event.clear()
        with self._lock:
            self.call_count = 0
class _BlockingChunked(Runnable[str, str]):
    """Streams chunks, blocking until released."""

    def __init__(self) -> None:
        self.stream_count = 0
        self._start_event = threading.Event()
        self._lock = threading.Lock()

    @property
    def InputType(self) -> type[str]:
        return str

    @property
    def OutputType(self) -> type[str]:
        return str

    def invoke(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> str:
        return input.upper()

    def stream(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Iterator[str]:
        with self._lock:
            self.stream_count += 1
        self._start_event.wait(timeout=10)
        for c in input:
            yield c.upper()

    async def astream(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        with self._lock:
            self.stream_count += 1
        while not self._start_event.is_set():
            await asyncio.sleep(0.01)
        for c in input:
            yield c.upper()

    def release(self) -> None:
        self._start_event.set()

    def reset(self) -> None:
        self._start_event.clear()
        with self._lock:
            self.stream_count = 0
class _Failing(Runnable[str, str]):
    """Blocks then raises ValueError."""

    def __init__(self) -> None:
        self.call_count = 0
        self._event = threading.Event()
        self._lock = threading.Lock()

    @property
    def InputType(self) -> type[str]:
        return str

    @property
    def OutputType(self) -> type[str]:
        return str

    def invoke(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> str:
        with self._lock:
            self.call_count += 1
        self._event.wait(timeout=10)
        msg = "deliberate failure"
        raise ValueError(msg)

    async def ainvoke(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> str:
        with self._lock:
            self.call_count += 1
        while not self._event.is_set():
            await asyncio.sleep(0.01)
        msg = "deliberate failure"
        raise ValueError(msg)

    def stream(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Iterator[str]:
        with self._lock:
            self.call_count += 1
        self._event.wait(timeout=10)
        msg = "deliberate stream failure"
        raise ValueError(msg)

    async def astream(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        with self._lock:
            self.call_count += 1
        while not self._event.is_set():
            await asyncio.sleep(0.01)
        msg = "deliberate async stream failure"
        raise ValueError(msg)
        yield  # noqa: unreachable

    def release(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event.clear()
        with self._lock:
            self.call_count = 0
class _Chunked(Runnable[str, str]):
    """Non-blocking chunked streamer for simple tests."""

    def __init__(self) -> None:
        self.invoke_count = 0
        self.stream_count = 0

    @property
    def InputType(self) -> type[str]:
        return str

    @property
    def OutputType(self) -> type[str]:
        return str

    def invoke(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> str:
        self.invoke_count += 1
        return input.upper()

    def stream(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Iterator[str]:
        self.stream_count += 1
        for c in input:
            yield c.upper()

    async def astream(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        self.stream_count += 1
        for c in input:
            yield c.upper()
def test_with_coalesce_returns_runnable() -> None:
    r = RunnableLambda(lambda x: x).with_coalesce()
    assert isinstance(r, Runnable)
def test_invoke_returns_correct_result() -> None:
    r = RunnableLambda(lambda x: x.upper()).with_coalesce()
    assert r.invoke("hello") == "HELLO"
def test_concurrent_invoke_coalescing() -> None:
    inner = _Blocking()
    coalesced = inner.with_coalesce()
    results: list[str | None] = [None] * 5
    barrier = threading.Barrier(5)

    def worker(idx: int) -> None:
        barrier.wait()
        results[idx] = coalesced.invoke("hello")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    inner.release()
    for t in threads:
        t.join(timeout=10)
    assert inner.call_count == 1
    assert all(r == "HELLO" for r in results)
def test_different_inputs_not_coalesced() -> None:
    inner = _Blocking()
    coalesced = inner.with_coalesce()
    results: dict[int, str] = {}
    barrier = threading.Barrier(2)

    def worker(idx: int, inp: str) -> None:
        barrier.wait()
        results[idx] = coalesced.invoke(inp)

    threads = [
        threading.Thread(target=worker, args=(0, "aaa")),
        threading.Thread(target=worker, args=(1, "bbb")),
    ]
    for t in threads:
        t.start()
    inner.release()
    for t in threads:
        t.join(timeout=10)
    assert inner.call_count == 2
    assert results[0] == "AAA"
    assert results[1] == "BBB"
def test_sequential_calls_not_coalesced() -> None:
    inner = _Chunked()
    coalesced = inner.with_coalesce()
    coalesced.invoke("hello")
    coalesced.invoke("hello")
    assert inner.invoke_count == 2
def test_coalescing_key_ignores_config() -> None:
    inner = _Blocking()
    coalesced = inner.with_coalesce()
    results: list[str | None] = [None, None]
    barrier = threading.Barrier(2)

    def worker(idx: int, cfg: RunnableConfig) -> None:
        barrier.wait()
        results[idx] = coalesced.invoke("hello", config=cfg)

    t1 = threading.Thread(
        target=worker, args=(0, {"metadata": {"a": 1}})
    )
    t2 = threading.Thread(
        target=worker, args=(1, {"metadata": {"b": 2}})
    )
    t1.start()
    t2.start()
    time.sleep(0.3)
    inner.release()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert inner.call_count == 1
    assert results[0] == results[1] == "HELLO"
def test_coalescing_key_ignores_kwargs() -> None:
    inner = _Blocking()
    coalesced = inner.with_coalesce()
    results: list[str | None] = [None, None]
    barrier = threading.Barrier(2)

    def worker(idx: int, **kw: Any) -> None:
        barrier.wait()
        results[idx] = coalesced.invoke("hello", **kw)

    t1 = threading.Thread(
        target=worker, args=(0,), kwargs={"extra_a": "value1"}
    )
    t2 = threading.Thread(
        target=worker, args=(1,), kwargs={"extra_b": "value2"}
    )
    t1.start()
    t2.start()
    time.sleep(0.3)
    inner.release()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert inner.call_count == 1
def test_dict_key_ordering_coalesces() -> None:
    call_count = 0
    gate = threading.Event()

    def fn(x: Any) -> str:
        nonlocal call_count
        call_count += 1
        gate.wait(timeout=10)
        return str(sorted(x.items()))

    coalesced = RunnableLambda(fn).with_coalesce()
    results: list[str | None] = [None, None]
    barrier = threading.Barrier(2)

    def worker(idx: int, inp: dict) -> None:
        barrier.wait()
        results[idx] = coalesced.invoke(inp)

    t1 = threading.Thread(
        target=worker, args=(0, {"a": 1, "b": 2})
    )
    t2 = threading.Thread(
        target=worker, args=(1, {"b": 2, "a": 1})
    )
    t1.start()
    t2.start()
    time.sleep(0.3)
    gate.set()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert call_count == 1
def test_stream_concurrent_callers_all_chunks() -> None:
    inner = _BlockingChunked()
    coalesced = inner.with_coalesce()
    results: dict[int, list[str]] = {}
    barrier = threading.Barrier(3)

    def worker(idx: int) -> None:
        barrier.wait()
        results[idx] = list(coalesced.stream("hi"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    inner.release()
    for t in threads:
        t.join(timeout=10)
    assert inner.stream_count == 1
    for idx in range(3):
        assert results[idx] == ["H", "I"]
def test_stream_late_joiner_gets_all_chunks() -> None:
    inner = _BlockingChunked()
    coalesced = inner.with_coalesce()
    results: dict[int, list[str]] = {}
    leader_started = threading.Event()

    def leader() -> None:
        leader_started.set()
        results[0] = list(coalesced.stream("hi"))

    def late_joiner() -> None:
        leader_started.wait(timeout=5)
        results[1] = list(coalesced.stream("hi"))

    t1 = threading.Thread(target=leader)
    t1.start()
    leader_started.wait(timeout=5)
    t2 = threading.Thread(target=late_joiner)
    t2.start()
    inner.release()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert inner.stream_count == 1
    assert results[0] == ["H", "I"]
    assert results[1] == ["H", "I"]
async def test_async_invoke_coalescing() -> None:
    inner = _Blocking()
    coalesced = inner.with_coalesce()

    async def caller() -> str:
        return await coalesced.ainvoke("hello")

    tasks = [asyncio.create_task(caller()) for _ in range(5)]
    await asyncio.sleep(0.3)
    inner.release()
    results = await asyncio.gather(*tasks)
    assert inner.call_count == 1
    assert all(r == "HELLO" for r in results)
async def test_async_stream_coalescing() -> None:
    inner = _BlockingChunked()
    coalesced = inner.with_coalesce()

    async def caller() -> list[str]:
        return [chunk async for chunk in coalesced.astream("hi")]

    tasks = [asyncio.create_task(caller()) for _ in range(3)]
    await asyncio.sleep(0.3)
    inner.release()
    results = await asyncio.gather(*tasks)
    assert inner.stream_count == 1
    for r in results:
        assert r == ["H", "I"]
def test_error_propagation_invoke() -> None:
    inner = _Failing()
    coalesced = inner.with_coalesce()
    errors: list[Exception | None] = [None] * 3
    barrier = threading.Barrier(3)

    def worker(idx: int) -> None:
        barrier.wait()
        try:
            coalesced.invoke("hello")
        except ValueError as e:
            errors[idx] = e

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    inner.release()
    for t in threads:
        t.join(timeout=10)
    assert inner.call_count == 1
    assert all(isinstance(e, ValueError) for e in errors)
def test_error_propagation_stream() -> None:
    inner = _Failing()
    coalesced = inner.with_coalesce()
    errors: list[Exception | None] = [None] * 3
    barrier = threading.Barrier(3)

    def worker(idx: int) -> None:
        barrier.wait()
        try:
            list(coalesced.stream("hello"))
        except ValueError as e:
            errors[idx] = e

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    inner.release()
    for t in threads:
        t.join(timeout=10)
    assert inner.call_count == 1
    assert all(isinstance(e, ValueError) for e in errors)
def test_batch_per_item_coalescing() -> None:
    inner = _Blocking()
    coalesced = inner.with_coalesce()
    results: list[list[str] | None] = [None]

    def do_batch() -> None:
        results[0] = coalesced.batch(["hello", "hello", "world"])

    t = threading.Thread(target=do_batch)
    t.start()
    inner.release()
    t.join(timeout=10)
    assert results[0] == ["HELLO", "HELLO", "WORLD"]
    assert inner.call_count == 2
def test_batch_preserves_order() -> None:
    inner = _Blocking()
    coalesced = inner.with_coalesce()
    results: list[list[str] | None] = [None]

    def do_batch() -> None:
        results[0] = coalesced.batch(["ccc", "aaa", "bbb", "aaa"])

    t = threading.Thread(target=do_batch)
    t.start()
    inner.release()
    t.join(timeout=10)
    assert results[0] == ["CCC", "AAA", "BBB", "AAA"]
def test_batch_empty_input() -> None:
    inner = _Chunked()
    coalesced = inner.with_coalesce()
    assert coalesced.batch([]) == []
def test_batch_as_completed_coalesced_yield_together() -> None:
    inner = _Blocking()
    coalesced = inner.with_coalesce()
    results: list[list[tuple[int, str]] | None] = [None]

    def do_batch() -> None:
        results[0] = list(
            coalesced.batch_as_completed(["hello", "world", "hello"])
        )

    t = threading.Thread(target=do_batch)
    t.start()
    inner.release()
    t.join(timeout=10)

    result_list = results[0]
    assert result_list is not None

    idx_results = {idx: r for idx, r in result_list}
    assert idx_results[0] == "HELLO"
    assert idx_results[1] == "WORLD"
    assert idx_results[2] == "HELLO"

    positions = {idx: pos for pos, (idx, _) in enumerate(result_list)}
    assert positions[2] == positions[0] + 1
async def test_abatch_per_item_coalescing() -> None:
    inner = _Blocking()
    coalesced = inner.with_coalesce()

    async def do_batch() -> list[str]:
        return await coalesced.abatch(["hello", "hello", "world"])

    task = asyncio.create_task(do_batch())
    await asyncio.sleep(0.2)
    inner.release()
    results = await task
    assert results == ["HELLO", "HELLO", "WORLD"]
    assert inner.call_count == 2
async def test_abatch_as_completed_coalescing() -> None:
    inner = _Blocking()
    coalesced = inner.with_coalesce()

    async def do_batch() -> list[tuple[int, str]]:
        return [
            (idx, r)
            async for idx, r in coalesced.abatch_as_completed(
                ["hello", "world", "hello"]
            )
        ]

    task = asyncio.create_task(do_batch())
    await asyncio.sleep(0.2)
    inner.release()
    result_list = await task

    idx_results = {idx: r for idx, r in result_list}
    assert idx_results[0] == "HELLO"
    assert idx_results[1] == "WORLD"
    assert idx_results[2] == "HELLO"

    positions = {idx: pos for pos, (idx, _) in enumerate(result_list)}
    assert positions[2] == positions[0] + 1
def test_transform_passthrough() -> None:
    inner = _Chunked()
    coalesced = inner.with_coalesce()
    chunks = list(coalesced.transform(iter(["hello"])))
    assert len(chunks) > 0
    stats = coalesced.coalesce_info()
    assert stats.total == 0
async def test_atransform_passthrough() -> None:
    inner = _Chunked()
    coalesced = inner.with_coalesce()

    async def async_input() -> AsyncIterator[str]:
        yield "hello"

    chunks = [chunk async for chunk in coalesced.atransform(async_input())]
    assert len(chunks) > 0
    stats = coalesced.coalesce_info()
    assert stats.total == 0
def test_callbacks_fire_for_joined_callers() -> None:
    gate = threading.Event()

    def fn(x: str) -> str:
        gate.wait(timeout=10)
        return x.upper()

    coalesced = RunnableLambda(fn).with_coalesce()
    handler1 = FakeCallbackHandler()
    handler2 = FakeCallbackHandler()
    barrier = threading.Barrier(2)

    def worker(handler: FakeCallbackHandler) -> None:
        barrier.wait()
        coalesced.invoke("hello", config={"callbacks": [handler]})

    t1 = threading.Thread(target=worker, args=(handler1,))
    t2 = threading.Thread(target=worker, args=(handler2,))
    t1.start()
    t2.start()
    time.sleep(0.3)
    gate.set()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert handler1.chain_starts >= 1
    assert handler1.chain_ends >= 1
    assert handler2.chain_starts >= 1
    assert handler2.chain_ends >= 1
def test_backend_protocol() -> None:
    backend = InMemoryCoalesceBackend()
    assert isinstance(backend, CoalesceBackend)
def test_backend_register_leader_joiner() -> None:
    backend = InMemoryCoalesceBackend()
    assert backend.register("key1") is True  # leader
    assert backend.register("key1") is False  # joiner
def test_backend_join_receives_result() -> None:
    backend = InMemoryCoalesceBackend()
    backend.register("key1")

    result_holder: list[str | None] = [None]

    def joiner() -> None:
        result_holder[0] = backend.join("key1")

    t = threading.Thread(target=joiner)
    t.start()
    backend.complete("key1", result="DONE")
    t.join(timeout=5)
    assert result_holder[0] == "DONE"
def test_backend_join_raises_on_error() -> None:
    backend = InMemoryCoalesceBackend()
    backend.register("key1")

    error_holder: list[Exception | None] = [None]

    def joiner() -> None:
        try:
            backend.join("key1")
        except ValueError as e:
            error_holder[0] = e

    t = threading.Thread(target=joiner)
    t.start()
    backend.complete("key1", error=ValueError("boom"))
    t.join(timeout=5)
    assert isinstance(error_holder[0], ValueError)
def test_backend_is_active() -> None:
    backend = InMemoryCoalesceBackend()
    assert backend.is_active("key1") is False
    backend.register("key1")
    assert backend.is_active("key1") is True
    backend.complete("key1", result="done")
    assert backend.is_active("key1") is False
async def test_backend_ais_active() -> None:
    backend = InMemoryCoalesceBackend()
    assert await backend.ais_active("key1") is False
    await backend.aregister("key1")
    assert await backend.ais_active("key1") is True
    await backend.acomplete("key1", result="done")
    assert await backend.ais_active("key1") is False
def test_stats_initial() -> None:
    backend = InMemoryCoalesceBackend()
    stats = backend.stats
    assert isinstance(stats, CoalesceStats)
    assert stats.active == 0
    assert stats.coalesced == 0
    assert stats.total == 0
def test_stats_after_operations() -> None:
    backend = InMemoryCoalesceBackend()
    backend.register("key1")  # leader => total=1, active=1
    assert backend.stats.total == 1
    assert backend.stats.active == 1
    assert backend.stats.coalesced == 0

    backend.register("key1")  # joiner => total=2, coalesced=1
    assert backend.stats.total == 2
    assert backend.stats.coalesced == 1
    assert backend.stats.active == 1

    backend.complete("key1", result="done")
    assert backend.stats.active == 0
    assert backend.stats.total == 2
    assert backend.stats.coalesced == 1
async def test_stats_cross_sync_async_visibility() -> None:
    backend = InMemoryCoalesceBackend()
    backend.register("sync_key")
    await backend.aregister("async_key")
    assert backend.stats.active == 2
    assert backend.stats.total == 2
    assert backend.is_active("sync_key")
    assert await backend.ais_active("async_key")
    backend.complete("sync_key", result="s")
    await backend.acomplete("async_key", result="a")
    assert backend.stats.active == 0
    assert backend.stats.total == 2
def test_coalesce_info() -> None:
    inner = RunnableLambda(lambda x: x)
    coalesced = inner.with_coalesce()
    info = coalesced.coalesce_info()
    assert isinstance(info, CoalesceStats)
    assert info.active == 0
    assert info.coalesced == 0
    assert info.total == 0
def test_coalesce_clear_resets_stats() -> None:
    inner = _Chunked()
    coalesced = inner.with_coalesce()
    coalesced.invoke("hello")
    coalesced.invoke("hello")
    info = coalesced.coalesce_info()
    assert info.total >= 2
    coalesced.coalesce_clear()
    info = coalesced.coalesce_info()
    assert info.active == 0
    assert info.coalesced == 0
    assert info.total == 0
async def test_coalesce_clear_cancels_waiters() -> None:
    inner = _Blocking()
    coalesced = inner.with_coalesce()

    error_holder: list[BaseException | None] = [None]

    async def joiner() -> None:
        try:
            await coalesced.ainvoke("hello")
        except asyncio.CancelledError as e:
            error_holder[0] = e

    task1 = asyncio.create_task(coalesced.ainvoke("hello"))
    await asyncio.sleep(0.2)
    task2 = asyncio.create_task(joiner())
    await asyncio.sleep(0.2)

    coalesced.coalesce_clear()
    inner.release()

    await asyncio.sleep(0.3)
    try:
        task1.result()
    except Exception:
        pass

    assert isinstance(error_holder[0], asyncio.CancelledError)
def test_coalesce_clear_cancels_sync_waiters() -> None:
    inner = _Blocking()
    coalesced = inner.with_coalesce()
    error_holder: list[BaseException | None] = [None]

    def joiner() -> None:
        try:
            coalesced.invoke("hello")
        except asyncio.CancelledError as e:
            error_holder[0] = e

    leader_t = threading.Thread(target=lambda: coalesced.invoke("hello"))
    leader_t.start()
    time.sleep(0.2)
    joiner_t = threading.Thread(target=joiner)
    joiner_t.start()
    time.sleep(0.2)

    coalesced.coalesce_clear()
    inner.release()

    leader_t.join(timeout=10)
    joiner_t.join(timeout=10)
    assert isinstance(error_holder[0], asyncio.CancelledError)
def test_graph_delegation() -> None:
    inner = _Chunked()
    coalesced = inner.with_coalesce()
    graph = coalesced.get_graph()
    inner_graph = inner.get_graph()
    assert len(graph.nodes) == len(inner_graph.nodes)
    assert len(graph.edges) == len(inner_graph.edges)
def test_graph_in_chain() -> None:
    chain_no_coalesce = (
        RunnableLambda(lambda x: x.strip())
        | RunnableLambda(lambda x: len(x))
    )
    chain_coalesced = (
        RunnableLambda(lambda x: x.strip())
        | RunnableLambda(lambda x: len(x)).with_coalesce()
    )
    g1 = chain_no_coalesce.get_graph()
    g2 = chain_coalesced.get_graph()
    assert len(g1.nodes) == len(g2.nodes)
    assert len(g1.edges) == len(g2.edges)
def test_separate_wrappers_independent() -> None:
    inner = _Chunked()
    c1 = inner.with_coalesce()
    c2 = inner.with_coalesce()

    c1.invoke("a")
    assert c1.coalesce_info().total >= 1
    assert c2.coalesce_info().total == 0
def test_shared_backend() -> None:
    backend = InMemoryCoalesceBackend()
    inner = _Blocking()
    c1 = inner.with_coalesce(backend=backend)
    c2 = inner.with_coalesce(backend=backend)
    results: list[str | None] = [None, None]
    barrier = threading.Barrier(2)

    def w1() -> None:
        barrier.wait()
        results[0] = c1.invoke("hello")

    def w2() -> None:
        barrier.wait()
        results[1] = c2.invoke("hello")

    t1 = threading.Thread(target=w1)
    t2 = threading.Thread(target=w2)
    t1.start()
    t2.start()
    time.sleep(0.3)
    inner.release()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert inner.call_count == 1
    assert results[0] == results[1] == "HELLO"
    assert backend.stats.coalesced >= 1
def test_thread_safety() -> None:
    inner = _Blocking()
    coalesced = inner.with_coalesce()
    num_threads = 20
    results: list[str | None] = [None] * num_threads
    barrier = threading.Barrier(num_threads)

    def worker(idx: int) -> None:
        barrier.wait()
        results[idx] = coalesced.invoke("hello")

    threads = [
        threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
    ]
    for t in threads:
        t.start()
    time.sleep(0.3)
    inner.release()
    for t in threads:
        t.join(timeout=10)
    assert inner.call_count == 1
    assert all(r == "HELLO" for r in results)
async def test_astream_events() -> None:
    coalesced = RunnableLambda(lambda x: f"ok-{x}").with_coalesce()
    events = [
        event async for event in coalesced.astream_events("hello", version="v2")
    ]
    event_types = [e["event"] for e in events]
    assert "on_chain_start" in event_types
    assert "on_chain_end" in event_types
async def test_astream_events_no_coalescing() -> None:
    call_count = 0

    def fn(x: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"ok-{x}"

    coalesced = RunnableLambda(fn).with_coalesce()

    async def collect_events() -> list[dict]:
        return [
            event
            async for event in coalesced.astream_events("hello", version="v2")
        ]

    tasks = [asyncio.create_task(collect_events()) for _ in range(3)]
    all_events = await asyncio.gather(*tasks)
    assert call_count == 3
    for events in all_events:
        event_types = [e["event"] for e in events]
        assert "on_chain_start" in event_types
        assert "on_chain_end" in event_types
def test_exports_from_runnables_init() -> None:
    import langchain_core.runnables as runnables_mod

    assert hasattr(runnables_mod, "CoalesceBackend")
    assert hasattr(runnables_mod, "CoalesceStats")
    assert hasattr(runnables_mod, "InMemoryCoalesceBackend")
    assert not hasattr(runnables_mod, "RunnableCoalesce")
def test_imports_from_coalesce_module() -> None:
    from langchain_core.runnables.coalesce import (  # noqa: F401
        CoalesceBackend,
        CoalesceStats,
        InMemoryCoalesceBackend,
    )
async def test_async_backend_register_join_complete() -> None:
    backend = InMemoryCoalesceBackend()
    assert await backend.aregister("key1") is True
    assert await backend.aregister("key1") is False

    result_holder: list[str | None] = [None]

    async def joiner() -> None:
        result_holder[0] = await backend.ajoin("key1")

    task = asyncio.create_task(joiner())
    await asyncio.sleep(0.1)
    await backend.acomplete("key1", result="ASYNC_DONE")
    await task
    assert result_holder[0] == "ASYNC_DONE"
async def test_async_backend_join_raises_on_error() -> None:
    backend = InMemoryCoalesceBackend()
    await backend.aregister("key1")

    async def joiner() -> None:
        await backend.ajoin("key1")

    task = asyncio.create_task(joiner())
    await asyncio.sleep(0.1)
    await backend.acomplete("key1", error=ValueError("async boom"))
    with pytest.raises(ValueError, match="async boom"):
        await task
def test_error_not_persisted() -> None:
    call_count = 0

    def sometimes_fail(x: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = "first call fails"
            raise ValueError(msg)
        return x.upper()

    coalesced = RunnableLambda(sometimes_fail).with_coalesce()
    with pytest.raises(ValueError):
        coalesced.invoke("hello")
    result = coalesced.invoke("hello")
    assert result == "HELLO"
    assert call_count == 2
async def test_async_error_propagation() -> None:
    inner = _Failing()
    coalesced = inner.with_coalesce()

    async def caller() -> str:
        return await coalesced.ainvoke("hello")

    tasks = [asyncio.create_task(caller()) for _ in range(3)]
    await asyncio.sleep(0.3)
    inner.release()

    for task in tasks:
        with pytest.raises(ValueError):
            await task
    assert inner.call_count == 1
def test_coalesce_clear_no_active() -> None:
    coalesced = RunnableLambda(lambda x: x).with_coalesce()
    coalesced.coalesce_clear()
    info = coalesced.coalesce_info()
    assert info.active == 0
    assert info.coalesced == 0
    assert info.total == 0
