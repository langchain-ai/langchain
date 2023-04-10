"""Tool for echoing exceptions during chain execution"""


from langchain.tools.base import BaseTool


class ExceptionTool(BaseTool):
    """Tool for echoing."""

    name: str = "Exception"
    description: str = (
        "Reports exceptions. Do not call"
    )

    def _run(self, query: str) -> str:
        """Echo the input."""
        return query

    async def _arun(self, query: str) -> str:
        return query
