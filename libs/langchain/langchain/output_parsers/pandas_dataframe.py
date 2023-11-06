import pandas as pd
import re
from typing import Any, Dict, List, Tuple
from langchain.pydantic_v1 import validator
from langchain.schema import BaseOutputParser, OutputParserException
from langchain.output_parsers.format_instructions import PANDAS_DATAFRAME_FORMAT_INSTRUCTIONS

class PandasDataFrameOutputParser(BaseOutputParser):
    """Parse an output using Pandas DataFrame format."""

    """The Pandas DataFrame to parse."""
    dataframe: Any

    @validator("dataframe")
    def validate_dataframe(cls, val):
        if issubclass(type(val), pd.DataFrame):
            return val
        if pd.DataFrame(val).empty:
            raise ValueError("DataFrame cannot be empty.")

        raise TypeError("Wrong type for 'dataframe', must be subclass of Pandas DataFrame (pd.DataFrame)")
    
    def parse_array(self, array: str, original_request_params: str) -> Tuple[List[int], str]:
        parsed_array = []

        # Check if the format is [1,3,5]
        if re.match(r'\[\d+(,\s*\d+)*\]', array):
            parsed_array = [int(i) for i in re.findall(r'\d+', array)]
        # Check if the format is [1..5]
        elif re.match(r'\[(\d+)\.\.(\d+)\]', array):
            match = re.match(r'\[(\d+)\.\.(\d+)\]', array)
            start, end = map(int, match.groups())
            parsed_array = list(range(start, end + 1))
        # Check if the format is ["column_name"]
        elif re.match(r'\[[a-zA-Z0-9_]+(?:,[a-zA-Z0-9_]+)*\]', array):
            match = re.match(r'\[[a-zA-Z0-9_]+(?:,[a-zA-Z0-9_]+)*\]', array)
            parsed_array = match.group().strip("[]").split(",")

        # Validate the array
        if parsed_array == []:
            raise OutputParserException(
                f"Request parameter '{original_request_params}' has an invalid array format. Please refer to the format instructions."
            )
        elif type(parsed_array[0]) == int and parsed_array[-1] > self.dataframe.index.max():
            raise OutputParserException(
                f"The specified maximum index {parsed_array[-1]} exceeds the maximum index of the Pandas DataFrame {self.dataframe.index.max()}"
            )

        return parsed_array, original_request_params.split('[')[0]

    def parse(self, request: str) -> Dict[str, Any]:
        splitted_request: Tuple[str, str] = request.strip().split(':')
        if len(splitted_request) != 2:
            raise OutputParserException(
                f"Request '{request}' is not correctly formatted. Please refer to the format instructions."
            )
        result = {}
        try:
            request_type, request_params = splitted_request
            match request_type:
                case 'Invalid column', 'Invalid operation':
                    raise OutputParserException(
                        f"{request}. Please refer to the format instructions."
                    )
                case 'column':
                    array_exists = re.search(r'(\[.*?\])', request_params)
                    if array_exists:
                        parsed_array, stripped_request_params = self.parse_array(array_exists.group(1), request_params)
                        filtered_df = self.dataframe[self.dataframe.index.isin(parsed_array)]
                        if len(parsed_array) == 1:
                            result[stripped_request_params] = filtered_df[stripped_request_params].iloc[parsed_array[0]]
                        else:
                            result[stripped_request_params] = filtered_df[stripped_request_params]
                    else:
                        result[request_params] = self.dataframe[request_params]
                case 'row':
                    array_exists = re.search(r'(\[.*?\])', request_params)
                    if array_exists:
                        parsed_array, stripped_request_params = self.parse_array(array_exists.group(1), request_params)
                        filtered_df = self.dataframe[self.dataframe.columns.intersection(parsed_array)]
                        if len(parsed_array) == 1:
                            result[stripped_request_params] = filtered_df.iloc[int(stripped_request_params)][parsed_array[0]]
                        else:
                            result[stripped_request_params] = filtered_df.iloc[int(stripped_request_params)]
                    else:
                        result[request_params] = self.dataframe.iloc[int(request_params)]
                case _:
                    array_exists = re.search(r'(\[.*?\])', request_params)
                    if array_exists:
                        parsed_array, stripped_request_params = self.parse_array(array_exists.group(1), request_params)
                        filtered_df = self.dataframe[self.dataframe.index.isin(parsed_array)]
                        result[request_type] = getattr(filtered_df[stripped_request_params], request_type)()
                    else:
                        result[request_type] = getattr(self.dataframe[request_params], request_type)()
        except AttributeError:
            raise OutputParserException(
                f"Request type '{request_type}' is possibly not supported. Please refer to the format instructions."
            )
        return result

    def get_format_instructions(self) -> str:
        return PANDAS_DATAFRAME_FORMAT_INSTRUCTIONS.format(columns=', '.join(self.dataframe.columns))
