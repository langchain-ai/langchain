"""Get fixtures for the database connection."""
import os
from contextlib import asynccontextmanager, contextmanager

import psycopg
from typing_extensions import AsyncGenerator, Generator

PG_USER = os.environ.get("PG_USER", "langchain")
PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "langchain")
PG_DATABASE = os.environ.get("PG_DATABASE", "langchain")

# Using a different port for testing than the default 5432
# to avoid conflicts with a running PostgreSQL instance
# This port matches the convention in langchain/docker/docker-compose.yml
# To spin up a PostgreSQL instance for testing, run:
# docker-compose -f docker/docker-compose.yml up -d postgres
PG_PORT = os.environ.get("PG_PORT", "6023")

DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"


@asynccontextmanager
async def asyncpg_client() -> AsyncGenerator[psycopg.AsyncConnection, None]:
    # Establish a connection to your test database
    conn = await psycopg.AsyncConnection.connect(conninfo=DSN)
    try:
        yield conn
    finally:
        # Cleanup: close the connection after the test is done
        await conn.close()


@contextmanager
def syncpg_client() -> Generator[psycopg.Connection, None, None]:
    # Establish a connection to your test database
    conn = psycopg.connect(conninfo=DSN)
    try:
        yield conn
    finally:
        # Cleanup: close the connection after the test is done
        conn.close()
