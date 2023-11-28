import logging
from typing import List, Optional

from langchain_core.pydantic_v1 import BaseModel, Extra

from langchain.chat_models.mlflow import ChatMlflow

logger = logging.getLogger(__name__)


# Ignoring type because below is valid pydantic code
# Unexpected keyword argument "extra" for "__init_subclass__" of "object"  [call-arg]
class ChatParams(BaseModel, extra=Extra.allow):  # type: ignore[call-arg]
    """Parameters for the `MLflow` LLM."""

    temperature: float = 0.0
    n: int = 1
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None


class ChatDatabricks(ChatMlflow):
    """`Databricks` chat models API.

    To use, you should have the ``mlflow`` python package installed.
    For more information, see https://mlflow.org/docs/latest/llms/deployments/databricks.html.

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatDatabricks

            chat = ChatDatabricks(
                target_uri="databricks",
                endpoint="chat",
                params={"temperature": 0.1},
            )
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "databricks-chat"

    @property
    def _extras(self):
        return ""
