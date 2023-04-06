import os

import pytest

# Getting the absolute path of the current file's directory
ABS_PATH = os.path.dirname(os.path.abspath(__file__))


# This fixture returns a string containing the path to the cassette directory for the
# current module
@pytest.fixture(scope="module")
def vcr_cassette_dir(request: pytest.FixtureRequest) -> str:
    return os.path.join(
        os.path.dirname(request.module.__file__),
        "cassettes",
        os.path.basename(request.module.__file__).replace(".py", ""),
    )


# This fixture returns a dictionary containing filter_headers options
# for replacing certain headers with dummy values during cassette playback
# Specifically, it replaces the authorization header with a dummy value to
# prevent sensitive data from being recorded in the cassette.
@pytest.fixture(scope="module")
def vcr_config() -> dict:
    return {
        "filter_headers": [
            ("authorization", "authorization-DUMMY"),
            ("X-OpenAI-Client-User-Agent", "X-OpenAI-Client-User-Agent-DUMMY"),
            ("User-Agent", "User-Agent-DUMMY"),
        ],
    }
