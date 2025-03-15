import logging
from urllib.parse import urlparse

from langchain_core._api import deprecated

from langchain_community.chat_models.mlflow import ChatMlflow

logger = logging.getLogger(__name__)


@deprecated(
    since="0.3.3",
    removal="1.0",
    alternative_import="databricks_langchain.ChatDatabricks",
)
class ChatDatabricks(ChatMlflow):
    """`Databricks` chat models API.

    To use, you should have the ``mlflow`` python package installed.
    For more information, see https://mlflow.org/docs/latest/llms/deployments.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatDatabricks

            chat_model = ChatDatabricks(
                target_uri="databricks",
                endpoint="databricks-llama-2-70b-chat",
                temperature=0.1,
            )

            # single input invocation
            print(chat_model.invoke("What is MLflow?").content)

            # single input invocation with streaming response
            for chunk in chat_model.stream("What is MLflow?"):
                print(chunk.content, end="|")
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

    def _validate_uri(self) -> None:
        if self.target_uri == "databricks":
            return

        if urlparse(self.target_uri).scheme != "databricks":
            raise ValueError(
                "Invalid target URI. The target URI must be a valid databricks URI."
            )
