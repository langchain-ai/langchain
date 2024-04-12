import json
from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.anysdk import AnySdkWrapper


class AnySdkAction(BaseTool):
    """Tool that queries the  AnySdk API."""

    api_wrapper: AnySdkWrapper = Field(default_factory=AnySdkWrapper)
    mode: str
    name: str = ""
    description: str = ""

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """Use the  AnySdk API to run an operation."""
        try:
            instruction_args = {**kwargs}
            params = json.dumps(instruction_args)
            return self.api_wrapper.run(self.mode, params)
        except (ValueError, KeyError):
            return "Invalid instructions format. \
                Expected a JSON object with 'action_input' field."
        
        
