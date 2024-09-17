"""Tool for the SceneXplain API."""

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.scenexplain import SceneXplainAPIWrapper


class SceneXplainInput(BaseModel):
    """Input for SceneXplain."""

    query: str = Field(..., description="The link to the image to explain")


class SceneXplainTool(BaseTool):
    """Tool that explains images."""

    name: str = "image_explainer"
    description: str = (
        "An Image Captioning Tool: Use this tool to generate a detailed caption "
        "for an image. The input can be an image file of any format, and "
        "the output will be a text description that covers every detail of the image."
    )
    api_wrapper: SceneXplainAPIWrapper = Field(default_factory=SceneXplainAPIWrapper)  # type: ignore[arg-type]

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)
