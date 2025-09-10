"""python scripts/update_mypy_ruff.py"""

import glob
import re
import subprocess
from pathlib import Path

# Ignoring errors since this script is run in a controlled environment
import toml  # type: ignore # pyright: ignore[reportMissingModuleSource]
import tomllib  # type: ignore # pyright: ignore[reportMissingImports]

ROOT_DIR = Path(__file__).parents[1]


def main():
    for path in glob.glob(str(ROOT_DIR / "libs/**/pyproject.toml"), recursive=True):
        if "libs/cli/" in path:
            continue
        print(path)
        with open(path, "rb") as f:
            pyproject = tomllib.load(f)
        try:
            pyproject["tool"]["poetry"]["group"]["typing"]["dependencies"]["mypy"] = (
                "^1.10"
            )
            pyproject["tool"]["poetry"]["group"]["lint"]["dependencies"]["ruff"] = (
                "^0.5"
            )
        except KeyError:
            continue
        with open(path, "w") as f:
            toml.dump(pyproject, f)
        cwd = "/".join(path.split("/")[:-1])

        subprocess.run(
            "poetry lock --no-update; poetry install --with lint; poetry run ruff format .; poetry run ruff --fix .",
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
        )

        completed = subprocess.run(
            "poetry lock --no-update; poetry install --with lint, typing; poetry run mypy . --no-color",
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
        )
        logs = completed.stdout.split("\n")

        to_ignore = {}
        for l in logs:
            match = re.match(r"^(.*):(\d+): error:.*\[(.*)\]", l)
            if match:
                path, line_no, error_type = match.groups()
                if (path, line_no) in to_ignore:
                    to_ignore[(path, line_no)].append(error_type)
                else:
                    to_ignore[(path, line_no)] = [error_type]
        print(len(to_ignore))
        for (error_path, line_no), error_types in to_ignore.items():
            all_errors = ", ".join(error_types)
            full_path = f"{cwd}/{error_path}"
            try:
                with open(full_path, "r") as f:
                    file_lines = f.readlines()
            except FileNotFoundError:
                continue
            file_lines[int(line_no) - 1] = (
                file_lines[int(line_no) - 1][:-1] + f"  # type: ignore[{all_errors}]\n"
            )
            with open(full_path, "w") as f:
                f.write("".join(file_lines))

        subprocess.run(
            "poetry lock --no-update; poetry install --with lint; poetry run ruff format .; poetry run ruff --fix .",
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    main()
