from __future__ import annotations

from typing import Iterator, List

from langchain.embeddings.databricks import DatabricksEmbeddings


def _chunk(texts: List[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(texts), size):
        yield texts[i : i + size]


class MlflowEmbeddings(DatabricksEmbeddings):
    """Wrapper around embeddings LLMs in MLflow.

    To use, you should have the `mlflow[genai]` python package installed.
    For more information, see https://mlflow.org/docs/latest/llms/deployments/server.html.

    Example:
        .. code-block:: python

            from langchain.embeddings import MlflowEmbeddings

            embeddings = MlflowEmbeddings(
                target_uri="<target_uri>",
                endpoint="<endpoint>"
            )
    """
