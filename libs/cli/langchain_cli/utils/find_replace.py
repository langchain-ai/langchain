"""Find and replace text in files."""

from pathlib import Path


def find_and_replace(source: str, replacements: dict[str, str]) -> str:
    """Find and replace text in a string.

    Args:
        source: The source string.
        replacements: A dictionary of `{find: replace}` pairs.

    Returns:
        The modified string.
    """
    rtn = source

    # replace keys in deterministic alphabetical order
    finds = sorted(replacements.keys())
    for find in finds:
        replace = replacements[find]
        rtn = rtn.replace(find, replace)
    return rtn


def replace_file(source: Path, replacements: dict[str, str]) -> None:
    """Replace text in a file."""
    try:
        content = source.read_text()
    except UnicodeDecodeError:
        # binary file
        return
    new_content = find_and_replace(content, replacements)
    if new_content != content:
        source.write_text(new_content)


def replace_glob(parent: Path, glob: str, replacements: dict[str, str]) -> None:
    """Replace text in files matching a glob pattern."""
    for file in parent.glob(glob):
        if not file.is_file():
            continue
        replace_file(file, replacements)
