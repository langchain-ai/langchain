from typing import Text, Optional

from langchain_community.utilities.mindsdb import BaseMindWrapper
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool


class AIMindTool(BaseTool):
    name: Text = "ai_mind"
    description: Text = (
        "A wrapper around [AI-Minds](https://mindscloud.ai/). "
        "Useful for when you need answers to questions from your data, stored in "
        "data sources including PostgreSQL, MySQL, MariaDB, ClickHouse, Snowflake, "
        "and Google BigQuery. "
        "Input should be a question in natural language."
    )
    api_wrapper: BaseMindWrapper

    def _run(
        self,
        query: Text,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Text:
        return self.api_wrapper.run(query)
