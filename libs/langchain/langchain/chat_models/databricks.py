import logging

from langchain.chat_models.mlflow import ChatMlflow

logger = logging.getLogger(__name__)


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

    target_uri: str = "databricks"
    """The target URI to use. Defaults to ``databricks``."""

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "databricks-chat"

    @property
    def _mlflow_extras(self) -> str:
        return ""
