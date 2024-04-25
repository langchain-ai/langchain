# ruff: noqa: E402
from __future__ import annotations

import pytest

pytest.importorskip("libcst")

import difflib
from pathlib import Path

from typer.testing import CliRunner

from langchain_cli.namespaces.migrate.main import app
from tests.unit_tests.migrate.cli_runner.cases import before, expected
from tests.unit_tests.migrate.cli_runner.folder import Folder


def find_issue(current: Folder, expected: Folder) -> str:
    for current_file, expected_file in zip(current.files, expected.files):
        if current_file != expected_file:
            if current_file.name != expected_file.name:
                return (
                    f"Files have "
                    f"different names: {current_file.name} != {expected_file.name}"
                )
            if isinstance(current_file, Folder) and isinstance(expected_file, Folder):
                return find_issue(current_file, expected_file)
            elif isinstance(current_file, Folder) or isinstance(expected_file, Folder):
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
                )
            )
    return "Unknown"


def test_command_line(tmp_path: Path) -> None:
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        before.create_structure(root=Path(td))
        # The input is used to force through the confirmation.
        result = runner.invoke(app, [before.name], input="y\n")
        assert result.exit_code == 0, result.output

        after = Folder.from_structure(Path(td) / before.name)

    assert after == expected, find_issue(after, expected)
