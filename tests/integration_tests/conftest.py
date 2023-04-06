import os

import pytest

ABS_PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def vcr_cassette_dir(request: pytest.FixtureRequest) -> str:
    return os.path.join(
        os.path.dirname(request.module.__file__),
        "cassettes",
        os.path.basename(request.module.__file__).replace(".py", ""),
    )


@pytest.fixture(scope="module")
def vcr_config() -> dict:
    return {
        "filter_headers": [("authorization", "DUMMY")],
    }
