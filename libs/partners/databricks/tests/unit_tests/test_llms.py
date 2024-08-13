"""Test Databricks Chat API wrapper."""

from langchain_databricks import DatabricksLLM


def test_initialization() -> None:
    """Test integration initialization."""
    DatabricksLLM()
