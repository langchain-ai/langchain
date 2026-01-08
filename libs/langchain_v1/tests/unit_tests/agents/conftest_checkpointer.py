from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver

from .memory_assert import (
    MemorySaverAssertImmutable,
)


@contextmanager
def _checkpointer_memory() -> Iterator[BaseCheckpointSaver[str]]:
    yield MemorySaverAssertImmutable()


@asynccontextmanager
async def _checkpointer_memory_aio() -> AsyncIterator[BaseCheckpointSaver[str]]:
    yield MemorySaverAssertImmutable()


# Placeholder functions for other checkpointer types that aren't available
@contextmanager
def _checkpointer_sqlite() -> Iterator[BaseCheckpointSaver[str]]:
    # Fallback to memory for now
    yield MemorySaverAssertImmutable()


@contextmanager
def _checkpointer_postgres() -> Iterator[BaseCheckpointSaver[str]]:
    # Fallback to memory for now
    yield MemorySaverAssertImmutable()


@contextmanager
def _checkpointer_postgres_pipe() -> Iterator[BaseCheckpointSaver[str]]:
    # Fallback to memory for now
    yield MemorySaverAssertImmutable()


@contextmanager
def _checkpointer_postgres_pool() -> Iterator[BaseCheckpointSaver[str]]:
    # Fallback to memory for now
    yield MemorySaverAssertImmutable()


@asynccontextmanager
async def _checkpointer_sqlite_aio() -> AsyncIterator[BaseCheckpointSaver[str]]:
    # Fallback to memory for now
    yield MemorySaverAssertImmutable()


@asynccontextmanager
async def _checkpointer_postgres_aio() -> AsyncIterator[BaseCheckpointSaver[str]]:
    # Fallback to memory for now
    yield MemorySaverAssertImmutable()


@asynccontextmanager
async def _checkpointer_postgres_aio_pipe() -> AsyncIterator[BaseCheckpointSaver[str]]:
    # Fallback to memory for now
    yield MemorySaverAssertImmutable()


@asynccontextmanager
async def _checkpointer_postgres_aio_pool() -> AsyncIterator[BaseCheckpointSaver[str]]:
    # Fallback to memory for now
    yield MemorySaverAssertImmutable()
