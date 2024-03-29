import re
from typing import Any, Dict, List, Tuple, Union

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.pydantic_v1 import validator

from langchain.output_parsers.format_instructions import (
    PANDAS_DATAFRAME_FORMAT_INSTRUCTIONS,
)


class PandasDataFrameOutputParser(BaseOutputParser):
    """Parse an output using Pandas DataFrame format."""

    """The Pandas DataFrame to parse."""
    dataframe: Any

    @validator("dataframe")
    def validate_dataframe(cls, val: Any) -> Any:
        import pandas as pd

        if issubclass(type(val), pd.DataFrame):
            return val
        if pd.DataFrame(val).empty:
            raise ValueError("DataFrame cannot be empty.")

        raise TypeError(
            "Wrong type for 'dataframe', must be a subclass \
                of Pandas DataFrame (pd.DataFrame)"
        )

    def parse_array(
        self, array: str, original_request_params: str
    ) -> Tuple[List[Union[int, str]], str]:
        parsed_array: List[Union[int, str]] = []

        # Check if the format is [1,3,5]
        if re.match(r"\[\d+(,\s*\d+)*\]", array):
            parsed_array = [int(i) for i in re.findall(r"\d+", array)]
        # Check if the format is [1..5]
        elif re.match(r"\[(\d+)\.\.(\d+)\]", array):
            match = re.match(r"\[(\d+)\.\.(\d+)\]", array)
            if match:
                start, end = map(int, match.groups())
                parsed_array = list(range(start, end + 1))
            else:
                raise OutputParserException(
                    f"Unable to parse the array provided in {array}. \
                        Please check the format instructions."
                )
        # Check if the format is ["column_name"]
        elif re.match(r"\[[a-zA-Z0-9_]+(?:,[a-zA-Z0-9_]+)*\]", array):
            match = re.match(r"\[[a-zA-Z0-9_]+(?:,[a-zA-Z0-9_]+)*\]", array)
            if match:
                parsed_array = list(map(str, match.group().strip("[]").split(",")))
            else:
                raise OutputParserException(
                    f"Unable to parse the array provided in {array}. \
                        Please check the format instructions."
                )

        # Validate the array
        if not parsed_array:
            raise OutputParserException(
                f"Invalid array format in '{original_request_params}'. \
                    Please check the format instructions."
            )
        elif (
            isinstance(parsed_array[0], int)
            and parsed_array[-1] > self.dataframe.index.max()
        ):
            raise OutputParserException(
                f"The maximum index {parsed_array[-1]} exceeds the maximum index of \
                    the Pandas DataFrame {self.dataframe.index.max()}."
            )

        return parsed_array, original_request_params.split("[")[0]

    def parse(self, request: str) -> Dict[str, Any]:
        stripped_request_params = None
        splitted_request = request.strip().split(":")
        if len(splitted_request) != 2:
            raise OutputParserException(
                f"Request '{request}' is not correctly formatted. \
                    Please refer to the format instructions."
            )
        result = {}
        try:
            request_type, request_params = splitted_request
            if request_type in {"Invalid column", "Invalid operation"}:
                raise OutputParserException(
                    f"{request}. Please check the format instructions."
                )
            array_exists = re.search(r"(\[.*?\])", request_params)
            if array_exists:
                parsed_array, stripped_request_params = self.parse_array(
                    array_exists.group(1), request_params
                )
                if request_type == "column":
                    filtered_df = self.dataframe[
                        self.dataframe.index.isin(parsed_array)
                    ]
                    if len(parsed_array) == 1:
                        result[stripped_request_params] = filtered_df[
                            stripped_request_params
                        ].iloc[parsed_array[0]]
                    else:
                        result[stripped_request_params] = filtered_df[
                            stripped_request_params
                        ]
                elif request_type == "row":
                    filtered_df = self.dataframe[
                        self.dataframe.columns.intersection(parsed_array)
                    ]
                    if len(parsed_array) == 1:
                        result[stripped_request_params] = filtered_df.iloc[
                            int(stripped_request_params)
                        ][parsed_array[0]]
                    else:
                        result[stripped_request_params] = filtered_df.iloc[
                            int(stripped_request_params)
                        ]
                else:
                    filtered_df = self.dataframe[
                        self.dataframe.index.isin(parsed_array)
                    ]
                    result[request_type] = getattr(
                        filtered_df[stripped_request_params], request_type
                    )()
            else:
                if request_type == "column":
                    result[request_params] = self.dataframe[request_params]
                elif request_type == "row":
                    result[request_params] = self.dataframe.iloc[int(request_params)]
                else:
                    result[request_type] = getattr(
                        self.dataframe[request_params], request_type
                    )()
        except (AttributeError, IndexError, KeyError):
            if request_type not in {"column", "row"}:
                raise OutputParserException(
                    f"Unsupported request type '{request_type}'. \
                        Please check the format instructions."
                )
            raise OutputParserException(
                f"""Requested index {
                    request_params
                    if stripped_request_params is None
                    else stripped_request_params
                } is out of bounds."""
            )

        return result

    def get_format_instructions(self) -> str:
        return PANDAS_DATAFRAME_FORMAT_INSTRUCTIONS.format(
            columns=", ".join(self.dataframe.columns)
        )
