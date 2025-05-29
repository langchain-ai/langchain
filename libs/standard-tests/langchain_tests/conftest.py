import base64
import gzip

import pytest
from vcr import VCR  # type: ignore[import-untyped]
from vcr.serializers import yamlserializer  # type: ignore[import-untyped]


class YamlGzipSerializer:
    @staticmethod
    def serialize(cassette_dict: dict) -> str:
        raw = yamlserializer.serialize(cassette_dict).encode("utf-8")
        compressed = gzip.compress(raw)
        return base64.b64encode(compressed).decode("ascii")

    @staticmethod
    def deserialize(data: str) -> dict:
        compressed = base64.b64decode(data.encode("ascii"))
        text = gzip.decompress(compressed).decode("utf-8")
        return yamlserializer.deserialize(text)


_BASE_FILTER_HEADERS = [
    ("authorization", "PLACEHOLDER"),
    ("x-api-key", "PLACEHOLDER"),
    ("api-key", "PLACEHOLDER"),
]


@pytest.fixture(scope="session")
def _base_vcr_config() -> dict:
    """
    Configuration that every cassette will receive.
    (Anything permitted by vcr.VCR(**kwargs) can be put here.)
    """
    return {
        "record_mode": "once",
        "filter_headers": _BASE_FILTER_HEADERS.copy(),
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
        "decode_compressed_response": True,
        "cassette_library_dir": "tests/cassettes",
        "path_transformer": VCR.ensure_suffix(".yaml"),
    }


@pytest.fixture(scope="session")
def vcr_config(_base_vcr_config: dict) -> dict:
    return _base_vcr_config
