"""Tool for the SceneXplain API."""

from pydantic import Field

from langchain.tools.base import BaseTool
from langchain.utilities.scenexplain import SceneXplainAPIWrapper


class SceneXplainTool(BaseTool):
    """Tool that adds the capability to explain images."""

    name = "Image Explainer"
    description = (
        "An Image Captioning Tool: Use this tool to generate a detailed caption "
        "for an image. The input can be an image file of any format, and "
        "the output will be a text description that covers every detail of the image."
    )
    api_wrapper: SceneXplainAPIWrapper = Field(default_factory=SceneXplainAPIWrapper)

    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SceneXplainTool does not support async")
