from typing import Text, Optional

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from langchain_community.utilities.mindsdb import DatabaseMindWrapper


class MindsDBTool(BaseTool):
    name: str = "mindsdb"
    description: Text = (

    )
    api_wrapper: DatabaseMindWrapper

    def _run(
        self,
        query: Text,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Text:
        return self.api_wrapper.run(query)
