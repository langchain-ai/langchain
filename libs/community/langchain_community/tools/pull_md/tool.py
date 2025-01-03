"""Tool for the Pull.md API."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.pull_md import PullMdAPIWrapper

class PullMdInput(BaseModel):
    """Input for the Pull.md tool."""
    url: str = Field(description="URL to convert to markdown")

class PullMdQueryRun(BaseTool):
    """Tool that uses Pull.md API to convert URLs to Markdown."""

    name: str = "pull_md"
    description: str = (
        "A wrapper around Pull.md service. "
        "Useful for converting web pages to Markdown format. "
        "Input should be a valid URL."
    )
    api_wrapper: PullMdAPIWrapper = Field(default_factory=PullMdAPIWrapper)
    args_schema: Type[BaseModel] = PullMdInput

    def _run(
        self,
        url: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Pull.md tool."""
        return self.api_wrapper.convert_url_to_markdown(url)
