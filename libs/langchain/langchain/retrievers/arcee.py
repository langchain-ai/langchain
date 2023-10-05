from typing import Any, Dict, Iterable, List, Optional

from langchain.pydantic_v1 import Extra, root_validator

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever
from langchain.schema.retriever import BaseRetriever
from langchain.utilities.arcee import ArceeWrapper, ArceeRoute, DALMFilter
from langchain.utils import get_from_dict_or_env


class ArceeRetriever(BaseRetriever):
    _client: ArceeWrapper = None  #: :meta private:
    """Arcee client."""

    arcee_api_key: str = ""
    """Arcee API Key"""

    model: str
    """Arcee DALM name"""

    arcee_api_url: str = "https://api.arcee.ai"
    """Arcee API URL"""

    arcee_api_version: str = "v2"
    """Arcee API Version"""

    arcee_app_url: str = "https://app.arcee.ai"
    """Arcee App URL"""

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        underscore_attrs_are_private = True

    def __init__(self, **data: Any) -> None:
        """Initializes private fields."""

        super().__init__(**data)

        self._client = ArceeWrapper(
            arcee_api_key=self.arcee_api_key,
            arcee_api_url=self.arcee_api_url,
            arcee_api_version=self.arcee_api_version,
            model_kwargs=self.model_kwargs,
            model_name=self.model,
        )

        self._client.validate_model_training_status()

    @root_validator()
    def validate_environments(cls, values: Dict) -> Dict:
        """Validate Arcee environment variables."""

        # validate env vars
        values["arcee_api_key"] = get_from_dict_or_env(
            values,
            "arcee_api_key",
            "ARCEE_API_KEY",
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
        if values.get("model_kwargs") is not None:
            kw = values.get("model_kwargs")

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
            qeury: Query to submit to the model
            size: The max number of context results to retrieve. Defaults to 3. (Can be less if filters are provided).
            filters: Filters to apply to the context dataset.
        """

        try:
            return self._client.retrieve(query=query, **kwargs)
        except Exception as e:
            raise ValueError(f"Error while retrieving documents: {e}") from e
