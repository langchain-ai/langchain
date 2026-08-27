from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore


@contextmanager
def _store_memory() -> Iterator[BaseStore]:
    yield InMemoryStore()


@asynccontextmanager
async def _store_memory_aio() -> AsyncIterator[BaseStore]:
    yield InMemoryStore()


# Placeholder functions for other store types that aren't available
@contextmanager
def _store_postgres() -> Iterator[BaseStore]:
    # Fallback to memory for now
    yield InMemoryStore()


@contextmanager
def _store_postgres_pipe() -> Iterator[BaseStore]:
    # Fallback to memory for now
    yield InMemoryStore()


@contextmanager
def _store_postgres_pool() -> Iterator[BaseStore]:
    # Fallback to memory for now
    yield InMemoryStore()


@asynccontextmanager
async def _store_postgres_aio() -> AsyncIterator[BaseStore]:
    # Fallback to memory for now
    yield InMemoryStore()


@asynccontextmanager
async def _store_postgres_aio_pipe() -> AsyncIterator[BaseStore]:
    # Fallback to memory for now
    yield InMemoryStore()


@asynccontextmanager
async def _store_postgres_aio_pool() -> AsyncIterator[BaseStore]:
    # Fallback to memory for now
    yield InMemoryStore()
