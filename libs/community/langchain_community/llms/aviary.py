import dataclasses
import os
from typing import Any, Dict, List, Mapping, Optional, Union, cast

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.utils import get_from_dict_or_env
from pydantic import ConfigDict, model_validator

from langchain_community.llms.utils import enforce_stop_tokens

TIMEOUT = 60


@dataclasses.dataclass
class AviaryBackend:
    """Aviary backend.

    Attributes:
        backend_url: The URL for the Aviary backend.
        bearer: The bearer token for the Aviary backend.
    """

    backend_url: str
    bearer: str

    def __post_init__(self) -> None:
        self.header = {"Authorization": self.bearer}

    @classmethod
    def from_env(cls) -> "AviaryBackend":
        aviary_url = os.getenv("AVIARY_URL")
        assert aviary_url, "AVIARY_URL must be set"

        aviary_token = os.getenv("AVIARY_TOKEN", "")

        bearer = f"Bearer {aviary_token}" if aviary_token else ""
        aviary_url += "/" if not aviary_url.endswith("/") else ""

        return cls(aviary_url, bearer)


def get_models() -> List[str]:
    """List available models"""
    backend = AviaryBackend.from_env()
    request_url = backend.backend_url + "-/routes"
    response = requests.get(request_url, headers=backend.header, timeout=TIMEOUT)
    try:
        result = response.json()
    except requests.JSONDecodeError as e:
        raise RuntimeError(
            f"Error decoding JSON from {request_url}. Text response: {response.text}"
        ) from e
    result = sorted(
        [k.lstrip("/").replace("--", "/") for k in result.keys() if "--" in k]
    )
    return result


def get_completions(
    model: str,
    prompt: str,
    use_prompt_format: bool = True,
    version: str = "",
) -> Dict[str, Union[str, float, int]]:
    """Get completions from Aviary models."""

    backend = AviaryBackend.from_env()
    url = backend.backend_url + model.replace("/", "--") + "/" + version + "query"
    response = requests.post(
        url,
        headers=backend.header,
        json={"prompt": prompt, "use_prompt_format": use_prompt_format},
        timeout=TIMEOUT,
    )
    try:
        return response.json()
    except requests.JSONDecodeError as e:
        raise RuntimeError(
            f"Error decoding JSON from {url}. Text response: {response.text}"
        ) from e


class Aviary(LLM):
    """Aviary hosted models.

    Aviary is a backend for hosted models. You can
    find out more about aviary at
    http://github.com/ray-project/aviary

    To get a list of the models supported on an
    aviary, follow the instructions on the website to
    install the aviary CLI and then use:
    `aviary models`

    AVIARY_URL and AVIARY_TOKEN environment variables must be set.

    Attributes:
        model: The name of the model to use. Defaults to "amazon/LightGPT".
        aviary_url: The URL for the Aviary backend. Defaults to None.
        aviary_token: The bearer token for the Aviary backend. Defaults to None.
        use_prompt_format: If True, the prompt template for the model will be ignored.
            Defaults to True.
        version: API version to use for Aviary. Defaults to None.

    Example:
        .. code-block:: python

            from langchain_community.llms import Aviary
            os.environ["AVIARY_URL"] = "<URL>"
            os.environ["AVIARY_TOKEN"] = "<TOKEN>"
            light = Aviary(model='amazon/LightGPT')
            output = light('How do you make fried rice?')
    """

    model: str = "amazon/LightGPT"
    aviary_url: Optional[str] = None
    aviary_token: Optional[str] = None
    # If True the prompt template for the model will be ignored.
    use_prompt_format: bool = True
    # API version to use for Aviary
    version: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and python package exists in environment."""
        aviary_url = get_from_dict_or_env(values, "aviary_url", "AVIARY_URL")
        aviary_token = get_from_dict_or_env(values, "aviary_token", "AVIARY_TOKEN")

        # Set env viarables for aviary sdk
        os.environ["AVIARY_URL"] = aviary_url
        os.environ["AVIARY_TOKEN"] = aviary_token

        try:
            aviary_models = get_models()
        except requests.exceptions.RequestException as e:
            raise ValueError(e)

        model = values.get("model")
        if model and model not in aviary_models:
            raise ValueError(f"{aviary_url} does not support model {values['model']}.")

        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model,
            "aviary_url": self.aviary_url,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return f"aviary-{self.model.replace('/', '-')}"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Aviary
        Args:
            prompt: The prompt to pass into the model.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = aviary("Tell me a joke.")
        """
        kwargs = {"use_prompt_format": self.use_prompt_format}
        if self.version:
            kwargs["version"] = self.version

        output = get_completions(
            model=self.model,
            prompt=prompt,
            **kwargs,
        )

        text = cast(str, output["generated_text"])
        if stop:
            text = enforce_stop_tokens(text, stop)

        return text
