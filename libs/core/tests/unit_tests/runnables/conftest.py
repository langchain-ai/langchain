"""Shared configuration for runnable unit tests."""

import os
from collections.abc import Iterator

import pytest
from langsmith.utils import get_env_var


@pytest.fixture(scope="session", autouse=True)
def _disable_local_langsmith_tracing() -> Iterator[None]:
    """Keep runnable unit tests independent of developer LangSmith env vars."""
    env_vars = (
        "LANGCHAIN_TESTS_USER_AGENT",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY",
        "LANGSMITH_LANGGRAPH_API_VARIANT",
        "LANGSMITH_PROJECT",
        "LANGSMITH_TRACING",
    )
    previous_env = {env_var: os.environ.get(env_var) for env_var in env_vars}
    os.environ["LANGSMITH_TRACING"] = "false"
    for env_var in env_vars:
        if env_var != "LANGSMITH_TRACING":
            os.environ.pop(env_var, None)
    if hasattr(get_env_var, "cache_clear"):
        get_env_var.cache_clear()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        for env_var, value in previous_env.items():
            if value is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = value
        if hasattr(get_env_var, "cache_clear"):
            get_env_var.cache_clear()  # type: ignore[attr-defined]
