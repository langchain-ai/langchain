from typing import Union, Optional
from pathlib import Path
from os import PathLike


class VwLogger:
    def __init__(self, path: Optional[Union[str, PathLike]]):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, vw_ex: str):
        if self.path:
            with open(self.path, "a") as f:
                f.write(f"{vw_ex}\n\n")

    def logging_enabled(self):
        return bool(self.path)
