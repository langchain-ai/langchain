from typing import Text, Optional

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from langchain_community.utilities.mindsdb import BaseMindWrapper


class AIMindTool(BaseTool):
    name: Text = "ai_mind"
    description: Text = (

    )
    api_wrapper: BaseMindWrapper

    def _run(
        self,
        query: Text,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Text:
        return self.api_wrapper.run(query)
