"""Unit tests for middleware imports."""


def test_model_response_import_from_middleware() -> None:
    """Test that ModelResponse can be imported from langchain.agents.middleware.

    This test verifies the fix for GitHub issue #33453 where ModelResponse
    was defined in types.py but not exported in __init__.py.
    """
    from langchain.agents.middleware import ModelResponse

    # Verify the class exists and has expected attributes
    assert ModelResponse is not None
    assert hasattr(ModelResponse, "__annotations__")

    # Verify it's a dataclass
    import dataclasses
    assert dataclasses.is_dataclass(ModelResponse)


def test_model_request_and_response_both_importable() -> None:
    """Test that both ModelRequest and ModelResponse can be imported together."""
    from langchain.agents.middleware import ModelRequest, ModelResponse

    # Verify both classes exist
    assert ModelRequest is not None
    assert ModelResponse is not None

    # Verify they're both dataclasses
    import dataclasses
    assert dataclasses.is_dataclass(ModelRequest)
    assert dataclasses.is_dataclass(ModelResponse)


def test_model_response_in_all() -> None:
    """Test that ModelResponse is in __all__ list."""
    import langchain.agents.middleware as middleware_module

    # Verify __all__ contains ModelResponse
    assert hasattr(middleware_module, "__all__")
    assert "ModelResponse" in middleware_module.__all__
    assert "ModelRequest" in middleware_module.__all__
