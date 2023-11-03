import pandas as pd
import re
from typing import Any, Dict, List, Tuple
from langchain.pydantic_v1 import validator
from langchain.schema import BaseOutputParser, OutputParserException

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
    
    def parse_array(self, array: str, original_request_params: str) -> List[int]:
        parsed_array = []

        # Check if the format is [1,3,5]
        if re.match(r'\[\d+(,\s*\d+)*\]', array):
            parsed_array = [int(i) for i in re.findall(r'\d+', array)]
        # Check if the format is [1..5]
        elif re.match(r'\[(\d+)\.\.(\d+)\]', array):
            match = re.match(r'\[(\d+)\.\.(\d+)\]', array)
            start, end = map(int, match.groups())
            parsed_array = list(range(start, end + 1))

        # Validate the array
        if parsed_array == []:
            raise OutputParserException(
                f"Request parameter '{original_request_params}' has an invalid array format. Please refer to the format instructions."
            )
        elif parsed_array[-1] > self.dataframe.index.max():
            raise OutputParserException(
                f"The specified maximum index {parsed_array[-1]} exceeds the maximum index of the Pandas DataFrame {self.dataframe.index.max()}"
            )

        return parsed_array, original_request_params.split('[')[0]

    # NOTE: LLM will use format instructions to generate query in correct format.
    #       parse() function will then take the output and apply it to the DataFrame
    def parse(self, request: str) -> Dict[str, Any]:
        splitted_request: Tuple[str, str] = request.strip().split(':')
        if len(splitted_request) != 2:
            raise OutputParserException(
                f"Request '{request}' is not correctly formatted. Please refer to the format instructions."
            )
        result = {}
        try:
            # NOTE: Can probably simplify using getattr(df, function_name)()
            # TODO: Implement data sanitization
            request_type, request_params = splitted_request
            match request_type:
                case 'column':
                    # TODO: Implement multiple column parsing
                    p_query = self.dataframe[request_params].to_string(header=False, index=False)
                    result[request_params] = p_query.split("\n")
                case 'row':
                    # TODO: Implement multiple row parsing
                    p_query = self.dataframe.iloc[int(request_params)].to_string(header=False, index=False)
                    result[request_params] = p_query.split("\n")
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
        return ""
