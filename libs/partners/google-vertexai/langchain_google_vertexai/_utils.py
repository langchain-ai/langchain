"""Utilities to init Vertex AI."""

import dataclasses
from importlib import metadata
from typing import Any, Callable, Dict, Optional, Union

import google.api_core
from google.api_core.gapic_v1.client_info import ClientInfo
from google.cloud import storage
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import create_base_retry_decorator
from vertexai.generative_models._generative_models import (  # type: ignore[import-untyped]
    Candidate,
)
from vertexai.language_models import (  # type: ignore[import-untyped]
    TextGenerationResponse,
)
from vertexai.preview.generative_models import Image  # type: ignore[import-untyped]


def create_retry_decorator(
    *,
    max_retries: int = 1,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Creates a retry decorator for Vertex / Palm LLMs."""

    errors = [
        google.api_core.exceptions.ResourceExhausted,
        google.api_core.exceptions.ServiceUnavailable,
        google.api_core.exceptions.Aborted,
        google.api_core.exceptions.DeadlineExceeded,
        google.api_core.exceptions.GoogleAPIError,
    ]
    decorator = create_base_retry_decorator(
        error_types=errors, max_retries=max_retries, run_manager=run_manager
    )
    return decorator


def raise_vertex_import_error(minimum_expected_version: str = "1.38.0") -> None:
    """Raise ImportError related to Vertex SDK being not available.

    Args:
        minimum_expected_version: The lowest expected version of the SDK.
    Raises:
        ImportError: an ImportError that mentions a required version of the SDK.
    """
    raise ImportError(
        "Please, install or upgrade the google-cloud-aiplatform library: "
        f"pip install google-cloud-aiplatform>={minimum_expected_version}"
    )


def get_client_info(module: Optional[str] = None) -> "ClientInfo":
    r"""Returns a custom user agent header.

    Args:
        module (Optional[str]):
            Optional. The module for a custom user agent header.
    Returns:
        google.api_core.gapic_v1.client_info.ClientInfo
    """
    langchain_version = metadata.version("langchain")
    client_library_version = (
        f"{langchain_version}-{module}" if module else langchain_version
    )
    return ClientInfo(
        client_library_version=client_library_version,
        user_agent=f"langchain/{client_library_version}",
    )


def load_image_from_gcs(path: str, project: Optional[str] = None) -> Image:
    """Loads im Image from GCS."""
    gcs_client = storage.Client(project=project)
    pieces = path.split("/")
    blobs = list(gcs_client.list_blobs(pieces[2], prefix="/".join(pieces[3:])))
    if len(blobs) > 1:
        raise ValueError(f"Found more than one candidate for {path}!")
    return Image.from_bytes(blobs[0].download_as_bytes())


def is_codey_model(model_name: str) -> bool:
    """Returns True if the model name is a Codey model."""
    return "code" in model_name


def is_gemini_model(model_name: str) -> bool:
    """Returns True if the model name is a Gemini model."""
    return model_name is not None and "gemini" in model_name


def get_generation_info(
    candidate: Union[TextGenerationResponse, Candidate],
    is_gemini: bool,
    *,
    stream: bool = False,
) -> Dict[str, Any]:
    if is_gemini:
        # https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini#response_body
        info = {
            "is_blocked": any([rating.blocked for rating in candidate.safety_ratings]),
            "safety_ratings": [
                {
                    "category": rating.category.name,
                    "probability_label": rating.probability.name,
                    "blocked": rating.blocked,
                }
                for rating in candidate.safety_ratings
            ],
            "citation_metadata": candidate.citation_metadata,
        }
    # https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text-chat#response_body
    else:
        info = dataclasses.asdict(candidate)
        info.pop("text")
        info = {k: v for k, v in info.items() if not k.startswith("_")}
    if stream:
        # Remove non-streamable types, like bools.
        info.pop("is_blocked")
    return info
