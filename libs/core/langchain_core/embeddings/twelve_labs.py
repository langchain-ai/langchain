import requests
from typing import Any, Optional

from langchain_core.documents.base import Blob
from langchain_core.embeddings.embeddings import EmbeddingModel, EmbeddingOutput
from langchain_core.pydantic_v1 import SecretStr, Field, root_validator
from langchain_core.runnables import RunnableConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from twelvelabs import TwelveLabs


class TwelveLabEmbedding(EmbeddingModel):
    """Twelve Lab embedding model."""

    api_key: SecretStr
    engine_name: str
    sleep_ms: int
    """Number of milliseconds to sleep between requests."""
    client: TwelveLabs = Field(default=None, exclude=True)  #: :meta private:

    @root_validator(pre=False, skip_on_failure=True)
    def post_init(self, values):
        from twelvelabs import TwelveLabs

        values["client"] = TwelveLabs(
            api_key=values["api_key"].get_secret_value(),
        )
        return values

    def _embed(
        self, input_: Blob, config: Optional[RunnableConfig], **kwargs: Any
    ) -> EmbeddingOutput:
        """Embed input."""
        if input_.mimetype == "text/plain":
            text = input_.as_string()

        # "Marengo-retrieval-2.6"

        text = input_.as_string()
        response = self.client.embed.create(
            engine_name=self.engine_name,
            text=text,
        )
        return EmbeddingOutput(embeddings=[])
