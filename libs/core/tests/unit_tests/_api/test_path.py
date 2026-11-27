from pathlib import Path

from langchain_core._api import path

HERE = Path(__file__).parent

ROOT = HERE.parent.parent.parent


def test_as_import_path() -> None:
    """Test that the path is converted to a LangChain import path."""
    # Verify that default paths are correct

    # if editable install, check directory structure
    if path.PACKAGE_DIR == ROOT / "langchain_core":
        assert path.PACKAGE_DIR == ROOT / "langchain_core"

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


def test_as_import_path_without_suffix(tmp_path: Path) -> None:
    """A file without a suffix keeps its full name in the import path."""
    file = tmp_path / "LICENSE"
    file.touch()

    assert path.as_import_path(file, relative_to=tmp_path) == "LICENSE"
