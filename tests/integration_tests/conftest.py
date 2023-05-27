import os
from pathlib import Path

import pytest

# Getting the absolute path of the current file's directory
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

# Getting the absolute path of the project's root directory
PROJECT_DIR = os.path.abspath(os.path.join(ABS_PATH, os.pardir, os.pardir))


# Loading the .env file if it exists
def _load_env() -> None:
    dotenv_path = os.path.join(PROJECT_DIR, "tests", "integration_tests", ".env")
    if os.path.exists(dotenv_path):
        from dotenv import load_dotenv

        load_dotenv(dotenv_path)


_load_env()


@pytest.fixture(scope="module")
def test_dir() -> Path:
    return Path(os.path.join(PROJECT_DIR, "tests", "integration_tests"))


# This fixture returns a string containing the path to the cassette directory for the
# current module
@pytest.fixture(scope="module")
def vcr_cassette_dir(request: pytest.FixtureRequest) -> str:
    return os.path.join(
        os.path.dirname(request.module.__file__),
        "cassettes",
        os.path.basename(request.module.__file__).replace(".py", ""),
    )
