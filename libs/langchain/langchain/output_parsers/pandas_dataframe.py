import pandas as pd
from typing import Any, Dict, List, Type
from langchain.pydantic_v1 import root_validator
from langchain.schema import BaseOutputParser, OutputParserException


class PandasDataFrameOutputParser(BaseOutputParser):
    """Parse an output that is one of a set of values."""

    dataframe: pd.DataFrame

    @property
    def _valid_values(self) -> List[str]:
        # TODO: Do we want something like this?
        return []

    def parse(self, request: str) -> Any:
        return

    def get_format_instructions(self) -> str:
        return ""
