from pydantic import BaseModel
from abc import abstractmethod, ABC

class BaseExample(BaseModel, ABC):
    """Base class for examples."""

    @abstractmethod
    @property
    def formatted(self) -> str:
        """Returns a formatted example as a string."""


class SimpleExample(BaseExample):

    text: str

    def formatted(self) -> str:
        return self.text