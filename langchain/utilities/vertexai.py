"""Utilities to init Vertex AI."""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from google.oauth2.service_account import Credentials


def _create_credentials_from_file(
    json_credentials_path: Optional[str],
) -> Optional["Credentials"]:
    """Creates credentials for GCP.

    Args:
        json_credentials_path: The path on the file system where the
        credentials are stored.

    Returns:
        An optional of Credentials or None, in which case the default
        will be used.
    """

    from google.oauth2 import service_account

    credentials = None
    if json_credentials_path is not None:
        credentials = service_account.Credentials.from_service_account_file(
            json_credentials_path
        )

    return credentials


def raise_vertex_import_error() -> ValueError:
    sdk = "google-cloud-aiplatform>=1.25.0"
    raise ImportError(
        "Could not import VertexAI. Please, install it with " f"pip install {sdk} "
    )


def init_vertexai(
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    staging_bucket: Optional[str] = None,
    credentials_path: Optional[str] = None,
) -> None:
    """Inits vertexai.

    Args:
        project: The default GCP project to use when making Vertex API calls.
        location: The default location to use when making API calls.
        staging_bucket: The default staging bucket to use to stage artifacts
            when making API calls.
        credentials_path: Tje local path to the json file with credentials.
    """
    credentials = _create_credentials_from_file(credentials_path)
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
