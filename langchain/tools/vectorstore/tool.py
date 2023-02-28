"""Tool for the VectorDBQA chain."""

from pydantic import root_validator

from langchain.chains.vector_db_qa.base import VectorDBQA
from langchain.tools.base import BaseTool


class VectorDBQATool(BaseTool):
    """Tool for the VectorDBQA chain. To be initialized with name and chain."""

    chain: VectorDBQA
    template: str = (
        "Useful for when you need to answer questions about {name}. "
        "Input should be a fully formed question."
    )
    description = ""

    @root_validator()
    def create_description_from_template(cls, values: dict) -> dict:
        """Create description from template."""
        if "name" in values:
            values["description"] = values["template"].format(name=values["name"])
        return values

    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.chain.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("VectorDBQATool does not support async")
