"""Tool for the SemanticScholar API."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper


class SemantscholarInput(BaseModel):
    """Input for the SemanticScholar tool."""

    query: str = Field(description="search query to look up")


class SemanticScholarQueryRun(BaseTool):
    """Tool that searches the semanticscholar API."""

    name: str = "semanticscholar"
    description: str = (
        "A wrapper around semantischolar.org "
        "Useful for when you need to answer to questions"
        "from research papers."
        "Input should be a search query."
    )
    api_wrapper: SemanticScholarAPIWrapper = Field(
        default_factory=SemanticScholarAPIWrapper  # type: ignore[arg-type]
    )
    args_schema: Type[BaseModel] = SemantscholarInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Semantic Scholar tool."""
        return self.api_wrapper.run(query)
