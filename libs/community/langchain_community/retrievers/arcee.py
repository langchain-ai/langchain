from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import SecretStr
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init

from langchain_community.utilities.arcee import ArceeWrapper, DALMFilter


class ArceeRetriever(BaseRetriever):
    """Arcee Domain Adapted Language Models (DALMs) retriever.

    To use, set the ``ARCEE_API_KEY`` environment variable with your Arcee API key,
    or pass ``arcee_api_key`` as a named parameter.

    Example:
        .. code-block:: python

            from langchain_community.retrievers import ArceeRetriever

            retriever = ArceeRetriever(
                model="DALM-PubMed",
                arcee_api_key="ARCEE-API-KEY"
            )

            documents = retriever.invoke("AI-driven music therapy")
    """

    _client: Optional[ArceeWrapper] = None  #: :meta private:
    """Arcee client."""

    arcee_api_key: SecretStr
    """Arcee API Key"""

    model: str
    """Arcee DALM name"""

    arcee_api_url: str = "https://api.arcee.ai"
    """Arcee API URL"""

    arcee_api_version: str = "v2"
    """Arcee API Version"""

    arcee_app_url: str = "https://app.arcee.ai"
    """Arcee App URL"""

    model_kwargs: Optional[Dict[str, Any]] = None
    """Keyword arguments to pass to the model."""

    class Config:
        extra = "forbid"
        underscore_attrs_are_private = True

    def __init__(self, **data: Any) -> None:
        """Initializes private fields."""

        super().__init__(**data)

        self._client = ArceeWrapper(
            arcee_api_key=self.arcee_api_key.get_secret_value(),
            arcee_api_url=self.arcee_api_url,
            arcee_api_version=self.arcee_api_version,
            model_kwargs=self.model_kwargs,
            model_name=self.model,
        )

        self._client.validate_model_training_status()

    @pre_init
    def validate_environments(cls, values: Dict) -> Dict:
        """Validate Arcee environment variables."""

        # validate env vars
        values["arcee_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "arcee_api_key",
                "ARCEE_API_KEY",
            )
        )

        values["arcee_api_url"] = get_from_dict_or_env(
            values,
            "arcee_api_url",
            "ARCEE_API_URL",
        )

        values["arcee_app_url"] = get_from_dict_or_env(
            values,
            "arcee_app_url",
            "ARCEE_APP_URL",
        )

        values["arcee_api_version"] = get_from_dict_or_env(
            values,
            "arcee_api_version",
            "ARCEE_API_VERSION",
        )

        # validate model kwargs
        if values["model_kwargs"]:
            kw = values["model_kwargs"]

            # validate size
            if kw.get("size") is not None:
                if not kw.get("size") >= 0:
                    raise ValueError("`size` must not be negative.")

            # validate filters
            if kw.get("filters") is not None:
                if not isinstance(kw.get("filters"), List):
                    raise ValueError("`filters` must be a list.")
                for f in kw.get("filters"):
                    DALMFilter(**f)

        return values

    def _get_relevant_documents(
        self, query: str, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        """Retrieve {size} contexts with your retriever for a given query

        Args:
            query: Query to submit to the model
            size: The max number of context results to retrieve.
            Defaults to 3. (Can be less if filters are provided).
            filters: Filters to apply to the context dataset.
        """

        try:
            if not self._client:
                raise ValueError("Client is not initialized.")
            return self._client.retrieve(query=query, **kwargs)
        except Exception as e:
            raise ValueError(f"Error while retrieving documents: {e}") from e
