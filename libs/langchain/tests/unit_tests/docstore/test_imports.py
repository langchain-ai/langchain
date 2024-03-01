from langchain import docstore
from tests.unit_tests import assert_all_importable

EXPECTED_ALL = ["DocstoreFn", "InMemoryDocstore", "Wikipedia"]


def test_all_imports() -> None:
    assert set(docstore.__all__) == set(EXPECTED_ALL)
    assert_all_importable(docstore)
