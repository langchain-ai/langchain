from __future__ import annotations


class File:
    def __init__(self, name: str, content: list[str] | None = None) -> None:
        self.name = name
        self.content = "\n".join(content or [])

    def __eq__(self, __value: object, /) -> bool:
        if not isinstance(__value, File):
            return NotImplemented

        if self.name != __value.name:
            return False

        return self.content == __value.content

    def __hash__(self) -> int:
        return hash((self.name, self.content))
