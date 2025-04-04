from pathlib import Path
from typing import Dict


def find_and_replace(source: str, replacements: Dict[str, str]) -> str:
    rtn = source

    # replace keys in deterministic alphabetical order
    finds = sorted(replacements.keys())
    for find in finds:
        replace = replacements[find]
        rtn = rtn.replace(find, replace)
    return rtn


def replace_file(source: Path, replacements: dict[str, str]) -> None:
    try:
        content = source.read_text()
    except UnicodeDecodeError:
        # binary file
        return
    new_content = find_and_replace(content, replacements)
    if new_content != content:
        source.write_text(new_content)


def replace_glob(parent: Path, glob: str, replacements: dict[str, str]) -> None:
    for file in parent.glob(glob):
        if not file.is_file():
            continue
        replace_file(file, replacements)
