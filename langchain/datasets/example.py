from dataclasses import dataclass
from dataset import LANGCHAIN_STOP_SEQUENCE


@dataclass
class Example:
    x: str
    y: str = ""
    x_prefix: str = ""
    y_prefix: str = ""
    stop_sequence: str = LANGCHAIN_STOP_SEQUENCE
