from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .file import File
    from .folder import Folder


@dataclass
class Case:
    source: Folder | File
    expected: Folder | File
    name: str
