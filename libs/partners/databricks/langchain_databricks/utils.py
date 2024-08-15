from typing import Any
from urllib.parse import urlparse


def get_deployment_client(target_uri: str) -> Any:
    if (target_uri != "databricks") and (urlparse(target_uri).scheme != "databricks"):
        raise ValueError(
            "Invalid target URI. The target URI must be a valid databricks URI."
        )

    try:
        from mlflow.deployments import get_deploy_client  # type: ignore[import-untyped]

        return get_deploy_client(target_uri)
    except ImportError as e:
        raise ImportError(
            "Failed to create the client. "
            "Please run `pip install mlflow` to install "
            "required dependencies."
        ) from e
