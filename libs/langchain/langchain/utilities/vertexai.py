"""Utilities to init Vertex AI."""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from google.auth.credentials import Credentials


def raise_vertex_import_error() -> None:
    """Raise ImportError related to Vertex SDK being not available.

    Raises:
        ImportError: an ImportError that mentions a required version of the SDK.
    """
    sdk = "'google-cloud-aiplatform>=1.26.1'"
    raise ImportError(
        "Could not import VertexAI. Please, install it with " f"pip install {sdk}"
    )


def init_vertexai(
    project: Optional[str] = None,
    location: Optional[str] = None,
    credentials: Optional["Credentials"] = None,
) -> None:
    """Init vertexai.

    Args:
        project: The default GCP project to use when making Vertex API calls.
        location: The default location to use when making API calls.
        credentials: The default custom
                credentials to use when making API calls. If not provided credentials
                will be ascertained from the environment.

    Raises:
        ImportError: If importing vertexai SDK did not succeed.
    """
    try:
        import vertexai
    except ImportError:
        raise_vertex_import_error()

    vertexai.init(
        project=project,
        location=location,
        credentials=credentials,
    )
