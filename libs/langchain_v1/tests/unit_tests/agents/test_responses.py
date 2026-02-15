"""Unit tests for langchain.agents.structured_output module."""

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from langchain.agents.structured_output import (
    OutputToolBinding,
    ProviderStrategy,
    ProviderStrategyBinding,
    ToolStrategy,
    _SchemaSpec,
)


class _TestModel(BaseModel):
    """A test model for structured output."""

    name: str
    age: int
    email: str = "default@example.com"


class CustomModel(BaseModel):
    """Custom model with a custom docstring."""

    value: float
    description: str


class EmptyDocModel(BaseModel):
    # No custom docstring, should have no description in tool
    data: str


class TestToolStrategy:
    """Test ToolStrategy dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic ToolStrategy creation."""
        strategy = ToolStrategy(schema=_TestModel)
        assert strategy.schema == _TestModel
        assert strategy.tool_message_content is None
        assert len(strategy.schema_specs) == 1
        assert strategy.schema_specs[0].schema == _TestModel

    def test_multiple_schemas(self) -> None:
        """Test ToolStrategy with multiple schemas."""
        strategy = ToolStrategy(schema=_TestModel | CustomModel)
        assert len(strategy.schema_specs) == 2
        assert strategy.schema_specs[0].schema == _TestModel
        assert strategy.schema_specs[1].schema == CustomModel

    def test_schema_with_tool_message_content(self) -> None:
        """Test ToolStrategy with tool message content."""
        strategy = ToolStrategy(schema=_TestModel, tool_message_content="custom message")
        assert strategy.schema == _TestModel
        assert strategy.tool_message_content == "custom message"
        assert len(strategy.schema_specs) == 1
        assert strategy.schema_specs[0].schema == _TestModel


class TestProviderStrategy:
    """Test ProviderStrategy dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic ProviderStrategy creation."""
        strategy = ProviderStrategy(schema=_TestModel)
        assert strategy.schema == _TestModel
        assert strategy.schema_spec.schema == _TestModel
        assert strategy.schema_spec.strict is None

    def test_strict(self) -> None:
        """Test ProviderStrategy creation with strict=True."""
        strategy = ProviderStrategy(schema=_TestModel, strict=True)
        assert strategy.schema == _TestModel
        assert strategy.schema_spec.schema == _TestModel
        assert strategy.schema_spec.strict is True

    def test_to_model_kwargs(self) -> None:
        strategy_default = ProviderStrategy(schema=_TestModel)
        assert strategy_default.to_model_kwargs() == {
            "response_format": {
                "json_schema": {
                    "name": "_TestModel",
                    "schema": {
                        "description": "A test model for structured output.",
                        "properties": {
                            "age": {"title": "Age", "type": "integer"},
                            "email": {
                                "default": "default@example.com",
                                "title": "Email",
                                "type": "string",
                            },
                            "name": {"title": "Name", "type": "string"},
                        },
                        "required": ["name", "age"],
                        "title": "_TestModel",
                        "type": "object",
                    },
                },
                "type": "json_schema",
            }
        }

    def test_to_model_kwargs_strict(self) -> None:
        strategy_default = ProviderStrategy(schema=_TestModel, strict=True)
        assert strategy_default.to_model_kwargs() == {
            "response_format": {
                "json_schema": {
                    "name": "_TestModel",
                    "schema": {
                        "description": "A test model for structured output.",
                        "properties": {
                            "age": {"title": "Age", "type": "integer"},
                            "email": {
                                "default": "default@example.com",
                                "title": "Email",
                                "type": "string",
                            },
                            "name": {"title": "Name", "type": "string"},
                        },
                        "required": ["name", "age"],
                        "title": "_TestModel",
                        "type": "object",
                    },
                    "strict": True,
                },
                "type": "json_schema",
            }
        }


class TestOutputToolBinding:
    """Test OutputToolBinding dataclass and its methods."""

    def test_from_schema_spec_basic(self) -> None:
        """Test basic OutputToolBinding creation from SchemaSpec."""
        schema_spec = _SchemaSpec(schema=_TestModel)
        tool_binding = OutputToolBinding.from_schema_spec(schema_spec)

        assert tool_binding.schema == _TestModel
        assert tool_binding.schema_kind == "pydantic"
        assert tool_binding.tool is not None
        assert tool_binding.tool.name == "_TestModel"

    def test_from_schema_spec_with_custom_name(self) -> None:
        """Test OutputToolBinding creation with custom name."""
        schema_spec = _SchemaSpec(schema=_TestModel, name="custom_tool_name")
        tool_binding = OutputToolBinding.from_schema_spec(schema_spec)
        assert tool_binding.tool.name == "custom_tool_name"

    def test_from_schema_spec_with_custom_description(self) -> None:
        """Test OutputToolBinding creation with custom description."""
        schema_spec = _SchemaSpec(schema=_TestModel, description="Custom tool description")
        tool_binding = OutputToolBinding.from_schema_spec(schema_spec)

        assert tool_binding.tool.description == "Custom tool description"

    def test_from_schema_spec_with_model_docstring(self) -> None:
        """Test OutputToolBinding creation using model docstring as description."""
        schema_spec = _SchemaSpec(schema=CustomModel)
        tool_binding = OutputToolBinding.from_schema_spec(schema_spec)

        assert tool_binding.tool.description == "Custom model with a custom docstring."

    def test_from_schema_spec_empty_docstring(self) -> None:
        """Test OutputToolBinding creation with model that has default docstring."""

        # Create a model with the same docstring as BaseModel
        class DefaultDocModel(BaseModel):
            # This should have the same docstring as BaseModel
            pass

        schema_spec = _SchemaSpec(schema=DefaultDocModel)
        tool_binding = OutputToolBinding.from_schema_spec(schema_spec)

        # Should use empty description when model has default BaseModel docstring
        assert not tool_binding.tool.description

    def test_parse_payload_pydantic_success(self) -> None:
        """Test successful parsing for Pydantic model."""
        schema_spec = _SchemaSpec(schema=_TestModel)
        tool_binding = OutputToolBinding.from_schema_spec(schema_spec)

        tool_args = {"name": "John", "age": 30}
        result = tool_binding.parse(tool_args)

        assert isinstance(result, _TestModel)
        assert result.name == "John"
        assert result.age == 30
        assert result.email == "default@example.com"  # default value

    def test_parse_payload_pydantic_validation_error(self) -> None:
        """Test parsing failure for invalid Pydantic data."""
        schema_spec = _SchemaSpec(schema=_TestModel)
        tool_binding = OutputToolBinding.from_schema_spec(schema_spec)

        # Missing required field 'name'
        tool_args = {"age": 30}

        with pytest.raises(ValueError, match="Failed to parse data to _TestModel"):
            tool_binding.parse(tool_args)


class TestProviderStrategyBinding:
    """Test ProviderStrategyBinding dataclass and its methods."""

    def test_from_schema_spec_basic(self) -> None:
        """Test basic ProviderStrategyBinding creation from SchemaSpec."""
        schema_spec = _SchemaSpec(schema=_TestModel)
        tool_binding = ProviderStrategyBinding.from_schema_spec(schema_spec)

        assert tool_binding.schema == _TestModel
        assert tool_binding.schema_kind == "pydantic"

    def test_parse_payload_pydantic_success(self) -> None:
        """Test successful parsing for Pydantic model."""
        schema_spec = _SchemaSpec(schema=_TestModel)
        tool_binding = ProviderStrategyBinding.from_schema_spec(schema_spec)

        message = AIMessage(content='{"name": "John", "age": 30}')
        result = tool_binding.parse(message)

        assert isinstance(result, _TestModel)
        assert result.name == "John"
        assert result.age == 30
        assert result.email == "default@example.com"  # default value

    def test_parse_payload_pydantic_validation_error(self) -> None:
        """Test parsing failure for invalid Pydantic data."""
        schema_spec = _SchemaSpec(schema=_TestModel)
        tool_binding = ProviderStrategyBinding.from_schema_spec(schema_spec)

        # Missing required field 'name'
        message = AIMessage(content='{"age": 30}')

        with pytest.raises(ValueError, match="Failed to parse data to _TestModel"):
            tool_binding.parse(message)

    def test_parse_payload_pydantic_json_error(self) -> None:
        """Test parsing failure for invalid JSON data."""
        schema_spec = _SchemaSpec(schema=_TestModel)
        tool_binding = ProviderStrategyBinding.from_schema_spec(schema_spec)

        message = AIMessage(content="invalid json")

        with pytest.raises(
            ValueError,
            match="Failed to parse structured output for _TestModel",
        ):
            tool_binding.parse(message)

    def test_parse_content_list(self) -> None:
        """Test successful parsing for Pydantic model with content as list."""
        schema_spec = _SchemaSpec(schema=_TestModel)
        tool_binding = ProviderStrategyBinding.from_schema_spec(schema_spec)

        message = AIMessage(
            content=['{"name":', {"content": ' "John",'}, {"type": "text", "text": ' "age": 30}'}]
        )
        result = tool_binding.parse(message)

        assert isinstance(result, _TestModel)
        assert result.name == "John"
        assert result.age == 30
        assert result.email == "default@example.com"  # default value


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_schema(self) -> None:
        """Test ToolStrategy with a single schema creates one schema spec."""
        strategy = ToolStrategy(EmptyDocModel)
        assert len(strategy.schema_specs) == 1

    def test_empty_docstring_model(self) -> None:
        """Test that models without explicit docstrings have empty tool descriptions."""
        binding = OutputToolBinding.from_schema_spec(_SchemaSpec(EmptyDocModel))
        assert binding.tool.name == "EmptyDocModel"
        assert not binding.tool.description
