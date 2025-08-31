from contextlib import asynccontextmanager, contextmanager

from langgraph.store.memory import InMemoryStore


@contextmanager
def _store_memory():
    store = InMemoryStore()
    yield store


@asynccontextmanager
async def _store_memory_aio():
    store = InMemoryStore()
    yield store


# Placeholder functions for other store types that aren't available
@contextmanager
def _store_postgres():
    # Fallback to memory for now
    store = InMemoryStore()
    yield store


@contextmanager
def _store_postgres_pipe():
    # Fallback to memory for now
    store = InMemoryStore()
    yield store


@contextmanager
def _store_postgres_pool():
    # Fallback to memory for now
    store = InMemoryStore()
    yield store


@asynccontextmanager
async def _store_postgres_aio():
    # Fallback to memory for now
    store = InMemoryStore()
    yield store


@asynccontextmanager
async def _store_postgres_aio_pipe():
    # Fallback to memory for now
    store = InMemoryStore()
    yield store


@asynccontextmanager
async def _store_postgres_aio_pool():
    # Fallback to memory for now
    store = InMemoryStore()
    yield store
