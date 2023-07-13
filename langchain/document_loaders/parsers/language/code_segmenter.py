from abc import ABC, abstractmethod
from typing import List


class CodeSegmenter(ABC):
    """The abstract class for the code segmenter."""

    def __init__(self, code: str):
        self.code = code

    def is_valid(self) -> bool:
        return True

    @abstractmethod
    def simplify_code(self) -> str:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def extract_functions_classes(self) -> List[str]:
        raise NotImplementedError  # pragma: no cover
