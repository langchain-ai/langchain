from typing import Any


class AnyStr(str):
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, str)
