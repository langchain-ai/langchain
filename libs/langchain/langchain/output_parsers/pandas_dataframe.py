import pandas as pd
from typing import Any, Dict, List, Tuple
from langchain.pydantic_v1 import root_validator
from langchain.schema import BaseOutputParser, OutputParserException


class PandasDataFrameOutputParser(BaseOutputParser):
    """Parse an output using Pandas DataFrame format."""

    """The Pandas DataFrame to parse."""
    dataframe: pd.DataFrame

    @root_validator()
    def check_dataframe(cls, values: Dict) -> Dict:
        # TODO: Might want to add more validation logic to this.
        dataframe = values.get('dataframe')
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame.")
        if dataframe.empty:
            raise ValueError("DataFrame cannot be empty.")
        return values

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
                    # Call getattr. If it fails, then except, otherwise, do the stuff.
                    result[request_type] = getattr(self.dataframe[request_params], request_type)()
        except AttributeError:
            raise OutputParserException(
                f"Request type '{request_type}' is possibly not supported. Please refer to the format instructions."
            )

    def get_format_instructions(self) -> str:
        return ""
