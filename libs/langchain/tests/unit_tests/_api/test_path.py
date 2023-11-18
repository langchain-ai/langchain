from pathlib import Path

from langchain._api import path

HERE = Path(__file__).parent

ROOT = HERE.parent.parent.parent


def test_as_import_path() -> None:
    """Test that the path is converted to a LangChain import path."""
    # Verify that default paths are correct
    assert path.PACKAGE_DIR == ROOT / "langchain"
    # Verify that as import path works correctly
    assert path.as_import_path(HERE, relative_to=ROOT) == "tests.unit_tests._api"
    assert (
        path.as_import_path(__file__, relative_to=ROOT)
        == "tests.unit_tests._api.test_path"
    )
    assert (
        path.as_import_path(__file__, suffix="create_agent", relative_to=ROOT)
        == "tests.unit_tests._api.test_path.create_agent"
    )
