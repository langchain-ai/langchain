from __future__ import annotations

import difflib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from langchain_cli.cli import app
from tests.unit_tests.migrate.cli_runner.cases import before, expected
from tests.unit_tests.migrate.cli_runner.folder import Folder

pytest.importorskip("gritql")


def find_issue(current: Folder, expected: Folder) -> str:
    for current_file, expected_file in zip(current.files, expected.files, strict=False):
        if current_file != expected_file:
            if current_file.name != expected_file.name:
                return (
                    f"Files have "
                    f"different names: {current_file.name} != {expected_file.name}"
                )
            if isinstance(current_file, Folder) and isinstance(expected_file, Folder):
                return find_issue(current_file, expected_file)
            if isinstance(current_file, Folder) or isinstance(expected_file, Folder):
                return (
                    f"One of the files is a "
                    f"folder: {current_file.name} != {expected_file.name}"
                )
            return "\n".join(
                difflib.unified_diff(
                    current_file.content.splitlines(),
                    expected_file.content.splitlines(),
                    fromfile=current_file.name,
                    tofile=expected_file.name,
                ),
            )
    return "Unknown"


@pytest.mark.xfail(reason="grit may not be installed in env")
def test_command_line(tmp_path: Path) -> None:
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        before.create_structure(root=Path(td))
        # The input is used to force through the confirmation.
        result = runner.invoke(app, ["migrate", before.name, "--force"])
        if result.exit_code != 0:
            raise RuntimeError(result.output)

        after = Folder.from_structure(Path(td) / before.name)

    if after != expected:
        raise ValueError(find_issue(after, expected))
