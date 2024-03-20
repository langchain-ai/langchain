import pytest

from langchain import storage
from tests.unit_tests import assert_all_importable

EXPECTED_ALL = [
    "EncoderBackedStore",
    "InMemoryStore",
    "InMemoryByteStore",
    "LocalFileStore",
    "create_lc_store",
    "create_kv_docstore",
]
EXPECTED_DEPRECATED_IMPORTS = [
    "UpstashRedisByteStore",
    "UpstashRedisStore",
    "RedisStore",
]


def test_all_imports() -> None:
    assert set(storage.__all__) == set(EXPECTED_ALL)
    assert_all_importable(storage)


def test_deprecated_imports() -> None:
    for import_ in EXPECTED_DEPRECATED_IMPORTS:
        with pytest.raises(ImportError) as e:
            getattr(storage, import_)
            assert "langchain_community" in e, f"{import_=} didn't error"
    with pytest.raises(AttributeError):
        getattr(storage, "foo")
