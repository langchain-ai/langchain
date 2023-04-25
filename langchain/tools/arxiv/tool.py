"Use arxiv api "

from typing import Any, Dict

from pydantic import BaseModel, Extra, root_validator
from langchain.tools.base import BaseTool
from langchain.tools.requests.tool import RequestsGetTool, TextRequestsWrapper

import arxiv
arxiv = ArxivAPIWrapper()

class ArxivQueryRun(BaseTool):
    """Tool that adds the capability to search using the Arxiv API."""

    name = "Arxiv"
    description = (
        "A wrapper around Arxiv. "
        "Useful for getting summary of articles from arxiv.org. "
        "Input should be a search query."
    )
    api_wrapper: ArxivAPIWrapper

    def _run(self, query: str) -> str:
        """Use the Arxiv tool."""
        return self.api_wrapper.run(query)

    async def _arun(self, query: str) -> str:
        """Use the Arxiv tool asynchronously."""
        raise NotImplementedError("ArxivAPIWrapper does not support async")



llm = OpenAI(temperature=0)
tools = [ArxivQueryRun(api_wrapper=arxiv)]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)


agent.run(" bayesian networks")
