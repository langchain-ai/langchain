from pathlib import Path

import pytest

# Getting the absolute path of the current file's directory
ABS_PATH = Path(__file__).resolve().parent

# Getting the absolute path of the project's root directory
PROJECT_DIR = ABS_PATH.parent.parent


# Loading the .env file if it exists
def _load_env() -> None:
    dotenv_path = PROJECT_DIR / "tests" / "integration_tests" / ".env"
    if dotenv_path.exists():
        from dotenv import load_dotenv

        load_dotenv(dotenv_path)


_load_env()


@pytest.fixture(scope="module")
def test_dir() -> Path:
    return PROJECT_DIR / "tests" / "integration_tests"


# This fixture returns a string containing the path to the cassette directory for the
# current module
@pytest.fixture(scope="module")
def vcr_cassette_dir(request: pytest.FixtureRequest) -> str:
    module = Path(request.module.__file__)
    return str(module.parent / "cassettes" / module.stem)
