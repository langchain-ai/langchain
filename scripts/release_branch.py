"""
python scripts/release_branch.py anthropic bagatur
"""

import glob
import subprocess
import sys

# Ignoring errors since this script is run in a controlled environment
import toml  # type: ignore # pyright: ignore[reportMissingModuleSource]
import tomllib  # type: ignore # pyright: ignore[reportMissingImports]


def main(*args):
    pkg = args[1]
    if len(args) >= 2:
        user = args[2]
    else:
        user = "auto"
    for path in glob.glob("./libs/**/pyproject.toml", recursive=True):
        if pkg in path:
            break

    with open(path, "rb") as f:
        pyproject = tomllib.load(f)
    major, minor, patch = pyproject["tool"]["poetry"]["version"].split(".")
    patch = str(int(patch) + 1)
    bumped = ".".join((major, minor, patch))
    pyproject["tool"]["poetry"]["version"] = bumped
    with open(path, "w") as f:
        toml.dump(pyproject, f)

    branch = f"{user}/{pkg}_{bumped.replace('.', '_')}"
    print(
        subprocess.run(
            f"git checkout -b {branch}; git commit -am '{pkg}[patch]: Release {bumped}'; git push -u origin {branch}",
            shell=True,
            capture_output=True,
            text=True,
        )
    )


if __name__ == "__main__":
    main(*sys.argv)
