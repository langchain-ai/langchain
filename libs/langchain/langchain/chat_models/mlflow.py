import logging
from typing import List, Optional

from langchain_core.pydantic_v1 import BaseModel, Extra

from langchain.llms.databricks import ChatDatabricks

logger = logging.getLogger(__name__)


# Ignoring type because below is valid pydantic code
# Unexpected keyword argument "extra" for "__init_subclass__" of "object"  [call-arg]
class ChatParams(BaseModel, extra=Extra.allow):  # type: ignore[call-arg]
    """Parameters for the `MLflow` LLM."""

    temperature: float = 0.0
    n: int = 1
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None


class ChatMlflow(ChatDatabricks):
    """`MLflow` chat models API.

    To use, you should have the `mlflow[genai]` python package installed.
    For more information, see https://mlflow.org/docs/latest/llms/deployments/server.html.

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatMlflow

            chat = ChatMlflow(
                target_uri="http://127.0.0.1:5000",
                endpoint="chat",
                params={
                    "temperature": 0.1
                }
            )
    """
