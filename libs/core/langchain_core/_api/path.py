import os
from pathlib import Path

HERE = Path(__file__).parent

# Get directory of langchain package
PACKAGE_DIR = HERE.parent
SEPARATOR = os.sep


def get_relative_path(file: Path | str, *, relative_to: Path = PACKAGE_DIR) -> str:
    """Get the path of the file as a relative path to the package directory.

    Args:
        file: The file path to convert.
        relative_to: The base path to make the file path relative to.

    Returns:
        The relative path as a string.
    """
    if isinstance(file, str):
        file = Path(file)
    return str(file.relative_to(relative_to))


def as_import_path(
    file: Path | str,
    *,
    suffix: str | None = None,
    relative_to: Path = PACKAGE_DIR,
) -> str:
    """Path of the file as a LangChain import exclude langchain top namespace.

    Args:
        file: The file path to convert.
        suffix: An optional suffix to append to the import path.
        relative_to: The base path to make the file path relative to.

    Returns:
        The import path as a string.
    """
    if isinstance(file, str):
        file = Path(file)
    path = get_relative_path(file, relative_to=relative_to)
    if file.is_file():
        path = path[: -len(file.suffix)]
    import_path = path.replace(SEPARATOR, ".")
    if suffix:
        import_path += "." + suffix
    return import_path
