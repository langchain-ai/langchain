import pytest

from langchain import docstore

EXPECTED_DEPRECATED_IMPORTS = ["DocstoreFn", "InMemoryDocstore", "Wikipedia"]


def test_deprecated_imports() -> None:
    for import_ in EXPECTED_DEPRECATED_IMPORTS:
        with pytest.raises(ImportError) as e:
            getattr(docstore, import_)
            assert "langchain_community" in e
    with pytest.raises(AttributeError):
        getattr(docstore, "foo")
