"""Test that imports work correctly."""

from langchain_meshtastic import __all__


def test_all_imports() -> None:
    """Test that all expected exports are available."""
    assert "MeshtasticSendTool" in __all__
    assert "MeshtasticSendInput" in __all__
    assert "__version__" in __all__


def test_tool_import() -> None:
    """Test that the tool can be imported."""
    from langchain_meshtastic import MeshtasticSendTool

    assert MeshtasticSendTool is not None


def test_input_import() -> None:
    """Test that the input schema can be imported."""
    from langchain_meshtastic import MeshtasticSendInput

    assert MeshtasticSendInput is not None
