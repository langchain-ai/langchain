from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseExample(BaseModel, ABC):
    """Base class for examples."""

    @property
    @abstractmethod
    def formatted(self) -> str:
        """Returns a formatted example as a string."""


class SimpleExample(BaseExample):

    text: str

    @property
    def formatted(self) -> str:
        return self.text


from typing import Sequence, Union


def convert_to_examples(
    examples: Sequence[Union[str, BaseExample]]
) -> Sequence[BaseExample]:
    new_examples = [
        example
        if isinstance(example, BaseExample)
        else SimpleExample(text=str(example))
        for example in examples
    ]
    return new_examples
