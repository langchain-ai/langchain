"""Utilities to init Vertex AI."""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from google.auth.credentials import Credentials


def raise_vertex_import_error(minimum_expected_version: str = "1.33.0") -> None:
    """Raise ImportError related to Vertex SDK being not available.

    Args:
        minimum_expected_version: The lowest expected version of the SDK.
    Raises:
        ImportError: an ImportError that mentions a required version of the SDK.
    """
    raise ImportError(
        "Could not import VertexAI. Please, install it with "
        f"pip install google-cloud-aiplatform>={minimum_expected_version}"
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
