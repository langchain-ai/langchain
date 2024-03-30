from typing import Dict, List

from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.nasa.prompt import (
    NASA_CAPTIONS_PROMPT,
    NASA_MANIFEST_PROMPT,
    NASA_METADATA_PROMPT,
    NASA_SEARCH_PROMPT,
)
from langchain_community.tools.nasa.tool import NasaAction
from langchain_community.utilities.nasa import NasaAPIWrapper


class NasaToolkit(BaseToolkit):
    """Nasa Toolkit."""

    tools: List[BaseTool] = []

    @classmethod
    def from_nasa_api_wrapper(cls, nasa_api_wrapper: NasaAPIWrapper) -> "NasaToolkit":
        operations: List[Dict] = [
            {
                "mode": "search_media",
                "name": "Search NASA Image and Video Library media",
                "description": NASA_SEARCH_PROMPT,
            },
            {
                "mode": "get_media_metadata_manifest",
                "name": "Get NASA Image and Video Library media metadata manifest",
                "description": NASA_MANIFEST_PROMPT,
            },
            {
                "mode": "get_media_metadata_location",
                "name": "Get NASA Image and Video Library media metadata location",
                "description": NASA_METADATA_PROMPT,
            },
            {
                "mode": "get_video_captions_location",
                "name": "Get NASA Image and Video Library video captions location",
                "description": NASA_CAPTIONS_PROMPT,
            },
        ]
        tools = [
            NasaAction(
                name=action["name"],
                description=action["description"],
                mode=action["mode"],
                api_wrapper=nasa_api_wrapper,
            )
            for action in operations
        ]
        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools
