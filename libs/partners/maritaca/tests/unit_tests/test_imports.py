"""Test imports for langchain-maritaca.

Author: Anderson Henrique da Silva
Location: Minas Gerais, Brasil
GitHub: https://github.com/anderson-ufrj
"""

from langchain_maritaca import ChatMaritaca, __version__


def test_chat_model_import() -> None:
    """Test that ChatMaritaca can be imported."""
    assert ChatMaritaca is not None


def test_version_import() -> None:
    """Test that version can be imported."""
    assert __version__ is not None
    assert isinstance(__version__, str)
