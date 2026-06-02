"""Tests for Pydantic 2.14+ compatibility fix (issue #37835).

This module contains regression tests to ensure that the __annotations__
invariant is preserved when creating subset models.

Issue: https://github.com/langchain-ai/langchain/issues/37835
Related PR: https://github.com/langchain-ai/langchain/pull/34537
"""

from typing import Any

import langchain.agents  # noqa: F401
from pydantic import BaseModel

from langchain_core.language_models.llms import BaseLLM, LLM
from langchain_core.utils.pydantic import _create_subset_model_v2


class TestSubsetModelAnnotationsInvariant:
    """Test that subset model creation preserves Python's __annotations__ invariant."""

    def test_subset_model_preserves_annotations_invariant(self) -> None:
        """Verify that __annotations__ follows Python's invariant after subset creation.

        Python 3.10+ invariant:
        getattr(cls, '__annotations__', {}) == cls.__dict__.get('__annotations__', {})

        This test ensures compatibility with Pydantic 2.14+.

        Regression test for: https://github.com/langchain-ai/langchain/issues/37835
        """

        class TestModel(BaseModel):
            field1: str
            field2: int
            field3: dict[str, Any]

        # Create subset model
        subset = _create_subset_model_v2("TestSubset", TestModel, ["field1", "field3"])

        # Check the Python invariant: both should return same thing
        getattr_result = getattr(subset, "__annotations__", {})
        dict_result = subset.__dict__.get("__annotations__", {})

        assert getattr_result == dict_result, (
            f"__annotations__ invariant violated: "
            f"getattr={getattr_result}, __dict__={dict_result}"
        )

        # Verify selected fields are in annotations
        assert "field1" in subset.__annotations__
        assert "field3" in subset.__annotations__
        assert "field2" not in subset.__annotations__

    def test_subset_model_with_optional_fields(self) -> None:
        """Test subset model creation with optional fields."""

        class TestModel(BaseModel):
            required_field: str
            optional_field: str | None = None
            another_field: int = 42

        subset = _create_subset_model_v2(
            "TestSubset", TestModel, ["required_field", "optional_field"]
        )

        # Verify invariant
        assert getattr(subset, "__annotations__", {}) == subset.__dict__.get(
            "__annotations__", {}
        )

        # Verify fields
        assert "required_field" in subset.__annotations__
        assert "optional_field" in subset.__annotations__
        assert "another_field" not in subset.__annotations__

    def test_subset_model_with_dict_annotation(self) -> None:
        """Test subset model creation with dict type annotations.

        This specifically tests the case that was breaking in Pydantic 2.14+.
        """

        class TestModel(BaseModel):
            config_dict: dict[str, Any]
            other_field: str

        subset = _create_subset_model_v2("TestSubset", TestModel, ["config_dict"])

        # Verify invariant
        assert getattr(subset, "__annotations__", {}) == subset.__dict__.get(
            "__annotations__", {}
        )

        # Verify dict annotation is properly preserved
        assert "config_dict" in subset.__annotations__
        assert "other_field" not in subset.__annotations__

    def test_subset_model_with_complex_annotations(self) -> None:
        """Test subset model with complex type annotations."""

        class TestModel(BaseModel):
            list_field: list[dict[str, Any]]
            union_field: str | int | None
            simple_field: str

        subset = _create_subset_model_v2(
            "TestSubset", TestModel, ["list_field", "union_field"]
        )

        # Verify invariant
        assert getattr(subset, "__annotations__", {}) == subset.__dict__.get(
            "__annotations__", {}
        )

        # Verify fields
        assert "list_field" in subset.__annotations__
        assert "union_field" in subset.__annotations__
        assert "simple_field" not in subset.__annotations__

    def test_subset_model_preserves_field_properties(self) -> None:
        """Test that field properties are preserved in subset model."""

        class TestModel(BaseModel):
            field_with_default: str = "default_value"
            field_without_default: int
            optional_field: str | None = None

        subset = _create_subset_model_v2(
            "TestSubset",
            TestModel,
            ["field_with_default", "field_without_default"],
        )

        # Verify invariant
        assert getattr(subset, "__annotations__", {}) == subset.__dict__.get(
            "__annotations__", {}
        )

        # Verify we can instantiate the model
        instance = subset(field_without_default=42)
        assert instance.field_with_default == "default_value"
        assert instance.field_without_default == 42

    def test_subset_model_with_descriptions(self) -> None:
        """Test subset model creation with field descriptions."""

        class TestModel(BaseModel):
            field1: str
            field2: int

        descriptions = {"field1": "First field", "field2": "Second field"}

        subset = _create_subset_model_v2(
            "TestSubset",
            TestModel,
            ["field1"],
            descriptions=descriptions,
        )

        # Verify invariant
        assert getattr(subset, "__annotations__", {}) == subset.__dict__.get(
            "__annotations__", {}
        )

        # Verify field
        assert "field1" in subset.__annotations__
        assert "field2" not in subset.__annotations__


class TestImportingAgentsWithBaseLLM:
    """Test the original reproduction case from issue #37835."""

    def test_importing_agents_does_not_break_basellm_construction(self) -> None:
        """Test that importing langchain.agents doesn't break BaseLLM construction.

        This was the original reproduction case for issue #37835:

        ```python
        import langchain.agents
        from langchain_core.language_models.llms import BaseLLM
        # TypeError: 'function' object is not subscriptable
        ```

        This test verifies the fix by successfully importing and constructing BaseLLM.
        """
        # If this doesn't raise, the bug is fixed
        assert BaseLLM is not None

        # Verify BaseLLM can be subclassed and instantiated
        class CustomLLM(LLM):

            @property
            def _llm_type(self) -> str:
                return "custom"

            def _call(self, prompt: str, **kwargs: Any) -> str:
                return "test response"

        llm = CustomLLM()
        assert llm._llm_type == "custom"


class TestSubsetModelEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_empty_subset(self) -> None:
        """Test creating a subset with no fields (edge case)."""

        class TestModel(BaseModel):
            field1: str
            field2: int

        # Creating subset with empty field list is technically valid but unusual
        # Pydantic should handle this gracefully
        subset = _create_subset_model_v2("TestSubset", TestModel, [])

        # Verify invariant holds even for empty subset
        assert getattr(subset, "__annotations__", {}) == subset.__dict__.get(
            "__annotations__", {}
        )

    def test_all_fields_subset(self) -> None:
        """Test creating a subset with all fields."""

        class TestModel(BaseModel):
            field1: str
            field2: int
            field3: float

        subset = _create_subset_model_v2(
            "TestSubset", TestModel, ["field1", "field2", "field3"]
        )

        # Verify invariant
        assert getattr(subset, "__annotations__", {}) == subset.__dict__.get(
            "__annotations__", {}
        )

        # Verify all fields are present
        assert len(subset.__annotations__) == 3
        assert "field1" in subset.__annotations__
        assert "field2" in subset.__annotations__
        assert "field3" in subset.__annotations__

    def test_subset_with_docstring(self) -> None:
        """Test subset model creation preserves docstring."""

        class TestModel(BaseModel):
            """Original model docstring."""

            field1: str
            field2: int

        custom_doc = "Custom subset model documentation"
        subset = _create_subset_model_v2(
            "TestSubset",
            TestModel,
            ["field1"],
            fn_description=custom_doc,
        )

        # Verify invariant
        assert getattr(subset, "__annotations__", {}) == subset.__dict__.get(
            "__annotations__", {}
        )

        # Verify docstring
        assert subset.__doc__ == custom_doc
