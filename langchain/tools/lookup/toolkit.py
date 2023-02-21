"""A lookup toolkit is a toolkit that assists an agent in looking up of tools."""


from typing import Any, List, Optional, Sequence

import numpy as np

from langchain.embeddings.base import Embeddings
from langchain.tools.base import BaseTool, BaseToolkit
from pydantic import Extra


class VectorBackedToolkit(BaseToolkit):

    """A lookup toolkit assists an agent in looking up of tools."""

    embeddings: Embeddings
    tools: Sequence[BaseTool]
    rec_threshold: float = 0.9  # Hack - very permissive
    embedding_table: np.ndarray

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def __init__(self, **data: Any) -> None:
        """Initialize toolkit."""
        if "tools" not in data:
            raise ValueError(f"tools must be provided to {self.__class__.__name__}.")
        tools = data["tools"]
        tool_reprs = [f"{tool.name}: {tool.description}" for tool in tools]
        if "embeddings" not in data:
            raise ValueError(
                f"embeddings must be provided to {self.__class__.__name__}."
            )
        all_embeds = data["embeddings"].embed_documents(tool_reprs)
        embedding_table = np.array(all_embeds)
        super().__init__(embedding_table=embedding_table, **data)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools

    def recommend_tool(self, query: str) -> Optional[BaseTool]:
        """Recommend a tool based on the given query."""
        query_embed = self.embeddings.embed_documents([query])[0]
        distances = np.linalg.norm(self.embedding_table - query_embed, axis=1)
        closest_index = np.argmin(distances)
        return (
            self.tools[closest_index]
            if distances[closest_index] < self.rec_threshold
            else None
        )
