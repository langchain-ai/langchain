"""Get fixtures for the database connection."""
import os
from contextlib import asynccontextmanager, contextmanager

import asyncpg
import psycopg2

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
async def asyncpg_client():
    # Establish a connection to your test database
    conn = await asyncpg.connect(dsn=DSN)
    try:
        yield conn
    finally:
        # Cleanup: close the connection after the test is done
        await conn.close()


@contextmanager
def syncpg_client():
    # Establish a connection to your test database
    conn = psycopg2.connect(dsn=DSN)
    try:
        yield conn
    finally:
        # Cleanup: close the connection after the test is done
        conn.close()
