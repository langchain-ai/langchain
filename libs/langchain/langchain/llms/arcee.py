from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.utilities.arcee import ArceeClient, ArceeRoute, DALMFilter
from langchain.llms.base import LLM
from langchain.pydantic_v1 import Extra, root_validator
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from langchain.utils import get_from_dict_or_env


class Arcee(LLM):
    """Arcee's Domain Adapted Language Models (DALMs).

    To use, set the ``ARCEE_API_KEY`` environment variable with your Arcee API key,
    or pass ``arcee_api_key`` as a named parameter.

    Example:
        .. code-block:: python

            from langchain.llms import Arcee

            arcee = Arcee(
                model="DPT-PubMed-7b",
                arcee_api_key="DUMMY-KEY"
            )

            response = arcee("Can?")
    """

    client: ArceeClient = None  #: :meta private:
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

    model_id: str = ""
    """Arcee Model ID"""

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "arcee"

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
                    raise ValueError("`size` must be positive")

            # validate filters
            if kw.get("filters") is not None:
                if not isinstance(kw.get("filters"), List):
                    raise ValueError("`filters` must be a list")
                for f in kw.get("filters"):
                    DALMFilter(**f)

        values["client"] = ArceeClient(
            arcee_api_key=values.get("arcee_api_key"),
            arcee_api_url=values.get("arcee_api_url"),
            arcee_api_version=values.get("arcee_api_version"),
            model_kwargs=values.get("model_kwargs"),
            model_name=values.get("model"),
        )

        # validate model training status
        values.get("client").validate_model_training_status()

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
            size: The max number of context results to retrieve. Defaults to 3. (Can be less if filters are provided).
            filters: Filters to apply to the context results.
        """

        try:
            response = self.client.make_request(
                method="post",
                route=ArceeRoute.generate.value.format(id_or_name=self.model_id),
                body=self.client.make_request_body_for_models(
                    prompt=prompt,
                    **kwargs,
                ),
            )
            return response["text"]
        except Exception as e:
            raise ValueError(f"Error while generating text: {e}") from e
