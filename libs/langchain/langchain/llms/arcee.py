from typing import Any, Dict, List, Optional, Union, cast

from langchain_core.pydantic_v1 import Extra, SecretStr, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.utilities.arcee import ArceeWrapper, DALMFilter
from langchain.utils import convert_to_secret_str, get_from_dict_or_env


class Arcee(LLM):
    """Arcee's Domain Adapted Language Models (DALMs).

    To use, set the ``ARCEE_API_KEY`` environment variable with your Arcee API key,
    or pass ``arcee_api_key`` as a named parameter.

    Example:
        .. code-block:: python

            from langchain.llms import Arcee

            arcee = Arcee(
                model="DALM-PubMed",
                arcee_api_key="ARCEE-API-KEY"
            )

            response = arcee("AI-driven music therapy")
    """

    _client: Optional[ArceeWrapper] = None  #: :meta private:
    """Arcee _client."""

    arcee_api_key: Union[SecretStr, str, None] = None
    """Arcee API Key"""

    model: str
    """Arcee DALM name"""

    arcee_api_url: str = "https://api.arcee.ai"
    """Arcee API URL"""

    arcee_api_version: str = "v2"
    """Arcee API Version"""

    arcee_app_url: str = "https://app.arcee.ai"
    """Arcee App URL"""

    model_id: str = ""
    """Arcee Model ID"""

    model_kwargs: Optional[Dict[str, Any]] = None
    """Keyword arguments to pass to the model."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        underscore_attrs_are_private = True

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "arcee"

    def __init__(self, **data: Any) -> None:
        """Initializes private fields."""

        super().__init__(**data)
        api_key = cast(SecretStr, self.arcee_api_key)
        self._client = ArceeWrapper(
            arcee_api_key=api_key,
            arcee_api_url=self.arcee_api_url,
            arcee_api_version=self.arcee_api_version,
            model_kwargs=self.model_kwargs,
            model_name=self.model,
        )

    @root_validator(pre=False)
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
        if values.get("model_kwargs"):
            kw = values["model_kwargs"]

            # validate size
            if kw.get("size") is not None:
                if not kw.get("size") >= 0:
                    raise ValueError("`size` must be positive")

            # validate filters
            if kw.get("filters") is not None:
                if not isinstance(kw.get("filters"), List):
                    raise ValueError("`filters` must be a list")
                for f in kw.get("filters"):
                    DALMFilter(**f)
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from Arcee DALM.

        Args:
            prompt: Prompt to generate text from.
            size: The max number of context results to retrieve.
            Defaults to 3. (Can be less if filters are provided).
            filters: Filters to apply to the context dataset.
        """

        try:
            if not self._client:
                raise ValueError("Client is not initialized.")
            return self._client.generate(prompt=prompt, **kwargs)
        except Exception as e:
            raise Exception(f"Failed to generate text: {e}") from e
