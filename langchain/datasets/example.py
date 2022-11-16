from abc import ABC
from dataclasses import dataclass
from dataset import LANGCHAIN_STOP_SEQUENCE


@dataclass
class Example(ABC):
    pass

@dataclass
class SimpleExample(Example):
    x: str
    y: str = ""
    x_prefix: str = ""
    y_prefix: str = ""
    stop_sequence: str = LANGCHAIN_STOP_SEQUENCE


@dataclass
class TemplateExample(Example):
    inputs: dict
    output: str = ""
    stop_sequence: str = LANGCHAIN_STOP_SEQUENCE

    def get_input_varaiables(self):
        return list(self.inputs.keys())
