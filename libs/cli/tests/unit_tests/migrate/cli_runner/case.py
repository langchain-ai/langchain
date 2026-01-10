from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.unit_tests.migrate.cli_runner.file import File
    from tests.unit_tests.migrate.cli_runner.folder import Folder


@dataclass
class Case:
    source: Folder | File
    expected: Folder | File
    name: str
