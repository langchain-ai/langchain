import pytest

_BASE_FILTER_HEADERS = [
    ("authorization", "PLACEHOLDER"),
    ("x-api-key", "PLACEHOLDER"),
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
    }


@pytest.fixture(scope="session")
def vcr_config(_base_vcr_config: dict) -> dict:
    return _base_vcr_config
