from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class PremAIEmbeddings(BaseModel, Embeddings):
    """Prem's Embedding APIs"""

    project_id: int
    """The project ID in which the experiments or deployments are carried out. 
    You can find all your projects here: https://app.premai.io/projects/"""

    premai_api_key: Optional[SecretStr] = None
    """Prem AI API Key. Get it here: https://app.premai.io/api_keys/"""

    model: str
    """The Embedding model to choose from"""

    show_progress_bar: bool = False
    """Whether to show a tqdm progress bar. Must have `tqdm` installed."""

    max_retries: int = 1
    """Max number of retries for tenacity"""

    client: Any

    @root_validator()
    def validate_environments(cls, values: Dict) -> Dict:
        """Validate that the package is installed and that the API token is valid"""
        try:
            from premai import Prem
        except ImportError as error:
            raise ImportError(
                "Could not import Prem Python package."
                "Please install it with: `pip install premai`"
            ) from error

        try:
            premai_api_key = get_from_dict_or_env(
                values, "premai_api_key", "PREMAI_API_KEY"
            )
            values["client"] = Prem(api_key=premai_api_key)
        except Exception as error:
            raise ValueError("Your API Key is incorrect. Please try again.") from error
        return values

    def embed_query(self, text: str) -> List[float]:
        """Embed query text"""
        embeddings = embed_with_retry(
            self, model=self.model, project_id=self.project_id, input=text
        )
        return embeddings.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = embed_with_retry(
            self, model=self.model, project_id=self.project_id, input=texts
        ).data

        return [embedding.embedding for embedding in embeddings]


def create_prem_retry_decorator(
    embedder: PremAIEmbeddings,
    *,
    max_retries: int = 1,
) -> Callable[[Any], Any]:
    """Create a retry decorator for PremAIEmbeddings.

    Args:
        embedder (PremAIEmbeddings): The PremAIEmbeddings instance
        max_retries (int): The maximum number of retries

    Returns:
        Callable[[Any], Any]: The retry decorator
    """
    import premai.models

    errors = [
        premai.models.api_response_validation_error.APIResponseValidationError,
        premai.models.conflict_error.ConflictError,
        premai.models.model_not_found_error.ModelNotFoundError,
        premai.models.permission_denied_error.PermissionDeniedError,
        premai.models.provider_api_connection_error.ProviderAPIConnectionError,
        premai.models.provider_api_status_error.ProviderAPIStatusError,
        premai.models.provider_api_timeout_error.ProviderAPITimeoutError,
        premai.models.provider_internal_server_error.ProviderInternalServerError,
        premai.models.provider_not_found_error.ProviderNotFoundError,
        premai.models.rate_limit_error.RateLimitError,
        premai.models.unprocessable_entity_error.UnprocessableEntityError,
        premai.models.validation_error.ValidationError,
    ]

    decorator = create_base_retry_decorator(
        error_types=errors, max_retries=max_retries, run_manager=None
    )
    return decorator


def embed_with_retry(
    embedder: PremAIEmbeddings,
    model: str,
    project_id: int,
    input: Union[str, List[str]],
) -> Any:
    """Using tenacity for retry in embedding calls"""
    retry_decorator = create_prem_retry_decorator(
        embedder, max_retries=embedder.max_retries
    )

    @retry_decorator
    def _embed_with_retry(
        embedder: PremAIEmbeddings,
        project_id: int,
        model: str,
        input: Union[str, List[str]],
    ) -> Any:
        embedding_response = embedder.client.embeddings.create(
            project_id=project_id, model=model, input=input
        )
        return embedding_response

    return _embed_with_retry(embedder, project_id=project_id, model=model, input=input)
