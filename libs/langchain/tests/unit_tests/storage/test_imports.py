from langchain import storage
from tests.unit_tests import assert_all_importable

EXPECTED_ALL = [
    "EncoderBackedStore",
    "InMemoryStore",
    "InMemoryByteStore",
    "LocalFileStore",
    "RedisStore",
    "create_lc_store",
    "create_kv_docstore",
    "UpstashRedisByteStore",
    "UpstashRedisStore",
]


def test_all_imports() -> None:
    assert set(storage.__all__) == set(EXPECTED_ALL)
    assert_all_importable(storage)
