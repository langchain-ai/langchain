"""Tests for structured output concurrency safety in langchain package.

These tests verify that the ProviderStrategyBinding and related structured
output mechanisms are safe under concurrent execution.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage
from langchain.agents.structured_output import (
    ProviderStrategyBinding,
    _SchemaSpec,
)


class TestOutputSchema(BaseModel):
    """Test schema for structured output validation."""

    category: str = Field(description="The category")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    items: list[str] = Field(default_factory=list, description="List of items")


class TestProviderStrategyBindingConcurrency:
    """Test suite for ProviderStrategyBinding concurrency safety."""

    def test_sequential_parsing_baseline(self) -> None:
        """Baseline test: sequential parsing should work correctly."""
        schema_spec = _SchemaSpec(TestOutputSchema)
        binding = ProviderStrategyBinding.from_schema_spec(schema_spec)

        message = AIMessage(
            content='{"category": "test", "confidence": 0.95, "items": ["a", "b"]}'
        )

        result = binding.parse(message)

        assert isinstance(result, TestOutputSchema)
        assert result.category == "test"
        assert result.confidence == 0.95
        assert result.items == ["a", "b"]

    def test_concurrent_parsing_isolation(self) -> None:
        """Test that concurrent parsing operations don't interfere."""
        schema_spec = _SchemaSpec(TestOutputSchema)
        binding = ProviderStrategyBinding.from_schema_spec(schema_spec)

        def parse_message(msg_id: int) -> TestOutputSchema:
            """Parse a message with unique data based on msg_id."""
            content = json.dumps(
                {
                    "category": f"category_{msg_id}",
                    "confidence": 0.5 + (msg_id % 5) * 0.1,
                    "items": [f"item_{msg_id}_1", f"item_{msg_id}_2"],
                }
            )
            message = AIMessage(content=content)
            return binding.parse(message)

        # Run 20 concurrent parsing operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(parse_message, i) for i in range(20)]
            results = [f.result() for f in futures]

        # Verify each result matches its expected data
        for i, result in enumerate(results):
            assert isinstance(result, TestOutputSchema)
            assert result.category == f"category_{i}"
            assert result.confidence == 0.5 + (i % 5) * 0.1
            assert result.items == [f"item_{i}_1", f"item_{i}_2"]

    def test_malformed_json_error_handling(self) -> None:
        """Test that malformed JSON produces clear error messages."""
        schema_spec = _SchemaSpec(TestOutputSchema)
        binding = ProviderStrategyBinding.from_schema_spec(schema_spec)

        # Malformed JSON - missing closing brace
        message = AIMessage(
            content='{"category": "test", "confidence": 0.95, "items": ["a"'
        )

        with pytest.raises(ValueError) as exc_info:
            binding.parse(message)

        error_msg = str(exc_info.value)
        assert "Failed to parse structured output" in error_msg
        assert "malformed data" in error_msg
        assert "JSON parsing error" in error_msg
        assert "concurrency" in error_msg.lower()

    def test_empty_content_error_handling(self) -> None:
        """Test that empty content produces clear error messages."""
        schema_spec = _SchemaSpec(TestOutputSchema)
        binding = ProviderStrategyBinding.from_schema_spec(schema_spec)

        # Empty content
        message = AIMessage(content="")

        with pytest.raises(ValueError) as exc_info:
            binding.parse(message)

        error_msg = str(exc_info.value)
        assert "empty content" in error_msg.lower()
        assert "incomplete streaming" in error_msg.lower()

    def test_whitespace_only_content_error_handling(self) -> None:
        """Test that whitespace-only content is handled correctly."""
        schema_spec = _SchemaSpec(TestOutputSchema)
        binding = ProviderStrategyBinding.from_schema_spec(schema_spec)

        # Whitespace only
        message = AIMessage(content="   \n\t  ")

        with pytest.raises(ValueError) as exc_info:
            binding.parse(message)

        error_msg = str(exc_info.value)
        assert "empty content" in error_msg.lower()

    def test_complex_content_extraction(self) -> None:
        """Test content extraction from complex message structures."""
        schema_spec = _SchemaSpec(TestOutputSchema)
        binding = ProviderStrategyBinding.from_schema_spec(schema_spec)

        # Message with list content containing text blocks
        message = AIMessage(
            content=[
                {"type": "text", "text": '{"category": "test1"'},
                {"type": "text", "text": ', "confidence": 0.8, "items": []}'},
            ]
        )

        result = binding.parse(message)

        assert isinstance(result, TestOutputSchema)
        assert result.category == "test1"
        assert result.confidence == 0.8
        assert result.items == []

    def test_concurrent_complex_parsing(self) -> None:
        """Test concurrent parsing with complex message structures."""
        schema_spec = _SchemaSpec(TestOutputSchema)
        binding = ProviderStrategyBinding.from_schema_spec(schema_spec)

        def parse_complex_message(msg_id: int) -> TestOutputSchema:
            """Parse a complex message structure."""
            # Alternate between string and list content
            if msg_id % 2 == 0:
                content: Any = json.dumps(
                    {
                        "category": f"cat_{msg_id}",
                        "confidence": 0.9,
                        "items": [f"x_{msg_id}"],
                    }
                )
            else:
                content = [
                    {
                        "type": "text",
                        "text": f'{{"category": "cat_{msg_id}", "confidence": 0.9, "items": ["x_{msg_id}"]}}',
                    }
                ]

            message = AIMessage(content=content)
            return binding.parse(message)

        # Run concurrent operations with mixed content types
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(parse_complex_message, i) for i in range(16)]
            results = [f.result() for f in futures]

        # Verify all results are correct
        for i, result in enumerate(results):
            assert result.category == f"cat_{i}"
            assert result.confidence == 0.9
            assert result.items == [f"x_{i}"]

    def test_json_schema_dict_parsing(self) -> None:
        """Test parsing with JSON schema dict instead of Pydantic model."""
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "number"},
            },
            "required": ["name", "value"],
        }

        schema_spec = _SchemaSpec(json_schema)
        binding = ProviderStrategyBinding.from_schema_spec(schema_spec)

        message = AIMessage(content='{"name": "test", "value": 42}')

        result = binding.parse(message)

        # JSON schema returns dict, not Pydantic model
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_concurrent_mixed_schema_types(self) -> None:
        """Test concurrent parsing with different schema types."""

        def parse_with_schema_type(use_pydantic: bool, idx: int) -> Any:
            """Parse using either Pydantic or JSON schema."""
            if use_pydantic:
                schema_spec = _SchemaSpec(TestOutputSchema)
                content = json.dumps(
                    {
                        "category": f"pyd_{idx}",
                        "confidence": 0.7,
                        "items": [],
                    }
                )
            else:
                json_schema = {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": "number"},
                    },
                }
                schema_spec = _SchemaSpec(json_schema)
                content = json.dumps({"name": f"json_{idx}", "value": idx})

            binding = ProviderStrategyBinding.from_schema_spec(schema_spec)
            message = AIMessage(content=content)
            return binding.parse(message)

        # Run concurrent operations with alternating schema types
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(parse_with_schema_type, i % 2 == 0, i)
                for i in range(12)
            ]
            results = [f.result() for f in futures]

        # Verify results based on schema type
        for i, result in enumerate(results):
            if i % 2 == 0:
                assert isinstance(result, TestOutputSchema)
                assert result.category == f"pyd_{i}"
            else:
                assert isinstance(result, dict)
                assert result["name"] == f"json_{i}"
                assert result["value"] == i
