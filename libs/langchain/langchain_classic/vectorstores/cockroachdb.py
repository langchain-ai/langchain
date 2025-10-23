"""CockroachDB vector store for LangChain.

This class uses the pgvector-compatible API provided by CockroachDB.
It inherits from the PGVector vector store to leverage existing functionality.
"""

from __future__ import annotations

from langchain_community.vectorstores.pgvector import PGVector


class CockroachDB(PGVector):
    """Vector store for CockroachDB using pgvector compatibility."""

    # CockroachDB exposes a pgvector-compatible API, so no overrides are necessary.
    pass

# Trigger CI
