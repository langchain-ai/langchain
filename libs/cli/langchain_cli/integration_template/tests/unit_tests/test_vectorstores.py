from __module_name__.vectorstores import (  # type: ignore[import-not-found]
    __ModuleName__VectorStore,  # type: ignore[import-not-found]
)


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    __ModuleName__VectorStore()
