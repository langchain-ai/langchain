from __future__ import annotations

from typing import Iterator, List

from langchain.embeddings.mlflow import MlflowEmbeddings


def _chunk(texts: List[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(texts), size):
        yield texts[i : i + size]


class DatabricksEmbeddings(MlflowEmbeddings):
    """Wrapper around embeddings LLMs in Databricks.

    To use, you should have the ``mlflow`` python package installed.
    For more information, see https://mlflow.org/docs/latest/llms/deployments/databricks.html.

    Example:
        .. code-block:: python

            from langchain.embeddings import DatabricksEmbeddings

            embeddings = DatabricksEmbeddings(
                target_uri="databricks",
                endpoint="embeddings",
            )
    """

    @property
    def _mlflow_extras(self) -> str:
        return ""
