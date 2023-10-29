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
    
    def parse_array(self, array: str) -> List[int]:
        # Check if the format is [1,3,5]
        if re.match(r'\[\d+(,\s*\d+)*\]', array):
            return [int(i) for i in re.findall(r'\d+', array)]
        # Check if the format is [1..5]
        elif re.match(r'\[(\d+)\.\.(\d+)\]', array):
            match = re.match(r'\[(\d+)\.\.(\d+)\]', array)
            start, end = map(int, match.groups())
            return list(range(start, end + 1))

        return []

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
            request_type, request_params = splitted_request
            match request_type:
                case 'column':
                    print(request_type)
                case 'row':
                    print(request_type)
                case _:
                    array_exists = re.search(r'(\[.*?\])', request_params)
                    if array_exists:
                        parsed_array = self.parse_array(array_exists.group(1))
                        if parsed_array == []:
                            raise OutputParserException(
                                "The array provided is not correctly defined. Please refer to the format instructions."
                            )
                    else:
                        result[request_type] = getattr(self.dataframe[request_params], request_type)()
        except AttributeError:
            raise OutputParserException(
                f"Request type '{request_type}' is possibly not supported. Please refer to the format instructions."
            )

    def get_format_instructions(self) -> str:
        return ""
