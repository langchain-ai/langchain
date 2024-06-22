from __future__ import annotations

from pathlib import Path

from .file import File


class Folder:
    def __init__(self, name: str, *files: Folder | File) -> None:
        self.name = name
        self._files = files

    @property
    def files(self) -> list[Folder | File]:
        return sorted(self._files, key=lambda f: f.name)

    def create_structure(self, root: Path) -> None:
        path = root / self.name
        path.mkdir()

        for file in self.files:
            if isinstance(file, Folder):
                file.create_structure(path)
            else:
                (path / file.name).write_text(file.content, encoding="utf-8")

    @classmethod
    def from_structure(cls, root: Path) -> Folder:
        name = root.name
        files: list[File | Folder] = []

        for path in root.iterdir():
            if path.is_dir():
                files.append(cls.from_structure(path))
            else:
                files.append(
                    File(path.name, path.read_text(encoding="utf-8").splitlines())
                )

        return Folder(name, *files)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, File):
            return False

        if not isinstance(__value, Folder):
            return NotImplemented

        if self.name != __value.name:
            return False

        if len(self.files) != len(__value.files):
            return False

        for self_file, other_file in zip(self.files, __value.files):
            if self_file != other_file:
                return False

        return True
