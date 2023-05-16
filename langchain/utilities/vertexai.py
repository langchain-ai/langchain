"""Utilities to init Vertex AI."""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from google.auth.credentials import Credentials


def raise_vertex_import_error() -> None:
    sdk = "'google-cloud-aiplatform>=1.25.0'"
    raise ImportError(
        "Could not import VertexAI. Please, install it with " f"pip install {sdk}"
    )


def init_vertexai(
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    staging_bucket: Optional[str] = None,
    credentials: Optional["Credentials"] = None,
) -> None:
    """Inits vertexai.

    Args:
        project: The default GCP project to use when making Vertex API calls.
        location: The default location to use when making API calls.
        staging_bucket: The default staging bucket to use to stage artifacts
            when making API calls.
        credentials (google.auth.credentials.Credentials): The default custom
                credentials to use when making API calls. If not provided credentials
                will be ascertained from the environment.
    """
    try:
        import vertexai

        vertexai.init(
            project=project_id,
            location=location,
            staging_bucket=staging_bucket,
            credentials=credentials,
        )
    except ImportError:
        raise_vertex_import_error


def is_tuned_model(model_name: str) -> bool:
    """Checks whether the model is a tuned one or not."""
    if model_name.startswith("projects/"):
        return True
    return False
