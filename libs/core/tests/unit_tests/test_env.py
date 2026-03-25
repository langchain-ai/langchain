"""Tests for env module."""

import platform

from langchain_core import __version__
from langchain_core.env import get_runtime_environment


def test_get_runtime_environment():
    """Test that get_runtime_environment returns expected keys."""
    env = get_runtime_environment()

    # Check required keys
    assert "library_version" in env
    assert "library" in env
    assert "platform" in env
    assert "runtime" in env
    assert "runtime_version" in env

    # Check values
    assert env["library_version"] == __version__
    assert env["library"] == "langchain-core"
    assert env["runtime"] == "python"
    assert env["runtime_version"] == platform.python_version()
    assert env["platform"] == platform.platform()


def test_get_runtime_environment_is_cached():
    """Test that get_runtime_environment uses lru_cache."""
    # Call twice and verify same object is returned (due to cache)
    env1 = get_runtime_environment()
    env2 = get_runtime_environment()
    assert env1 is env2
