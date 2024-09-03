"""Test MLX Chat wrapper."""

from importlib import import_module


def test_import_class() -> None:
    """Test that the class can be imported."""
    module_name = "langchain_community.chat_models.mlx"
    class_name = "ChatMLX"

    module = import_module(module_name)
    assert hasattr(module, class_name)
