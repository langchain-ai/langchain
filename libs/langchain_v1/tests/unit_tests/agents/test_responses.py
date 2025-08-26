"""Unit tests for langgraph.prebuilt.responses module."""

import pytest

# Skip this test since langgraph.prebuilt.responses is not available
pytest.skip("langgraph.prebuilt.responses not available", allow_module_level=True)


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


class TestUsingToolStrategy:
    """Test UsingToolStrategy dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic UsingToolStrategy creation."""
        strategy = ToolOutput(schema=_TestModel)
        assert strategy.schema == _TestModel
        assert strategy.tool_message_content is None
        assert len(strategy.schema_specs) == 1

    def test_multiple_schemas(self) -> None:
        """Test UsingToolStrategy with multiple schemas."""
        strategy = ToolOutput(schema=Union[_TestModel, CustomModel])
        assert len(strategy.schema_specs) == 2
        assert strategy.schema_specs[0].schema == _TestModel
        assert strategy.schema_specs[1].schema == CustomModel

    def test_schema_with_tool_message_content(self) -> None:
        """Test UsingToolStrategy with tool message content."""
        strategy = ToolOutput(schema=_TestModel, tool_message_content="custom message")
        assert strategy.schema == _TestModel
        assert strategy.tool_message_content == "custom message"
        assert len(strategy.schema_specs) == 1


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

    @pytest.mark.skip(reason="Need to fix bug in langchain-core for inheritance of doc-strings.")
    def test_from_schema_spec_empty_docstring(self) -> None:
        """Test OutputToolBinding creation with model that has default docstring."""

        # Create a model with the same docstring as BaseModel
        class DefaultDocModel(BaseModel):
            # This should have the same docstring as BaseModel
            pass

        schema_spec = _SchemaSpec(schema=DefaultDocModel)
        tool_binding = OutputToolBinding.from_schema_spec(schema_spec)

        # Should use empty description when model has default BaseModel docstring
        assert tool_binding.tool.description == ""

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


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_schemas_list(self) -> None:
        """Test UsingToolStrategy with empty schemas list."""
        strategy = ToolOutput(EmptyDocModel)
        assert len(strategy.schema_specs) == 1

    @pytest.mark.skip(reason="Need to fix bug in langchain-core for inheritance of doc-strings.")
    def test_base_model_doc_constant(self) -> None:
        """Test that BASE_MODEL_DOC constant is set correctly."""
        binding = OutputToolBinding.from_schema_spec(_SchemaSpec(EmptyDocModel))
        assert binding.tool.name == "EmptyDocModel"
        assert binding.tool.description[:5] == ""  # Should be empty for default docstring
