"""Wrapper around EdenAI's Generation API."""
import logging
from typing import Any, Dict, List, Literal, Optional

from aiohttp import ClientSession

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.pydantic_v1 import Extra, Field, root_validator
from langchain.requests import Requests
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class EdenAI(LLM):
    """Wrapper around edenai models.

    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings

    `feature` and `subfeature` are required, but any other model parameters can also be
    passed in with the format params={model_param: value, ...}

    for api reference check edenai documentation: http://docs.edenai.co.
    """

    base_url: str = "https://api.edenai.run/v2"

    edenai_api_key: Optional[str] = None

    feature: Literal["text", "image"] = "text"
    """Which generative feature to use, use text by default"""

    subfeature: Literal["generation"] = "generation"
    """Subfeature of above feature, use generation by default"""

    provider: str
    """Geneerative provider to use (eg: openai,stabilityai,cohere,google etc.)"""

    params: Dict[str, Any]
    """
    Parameters to pass to above subfeature (excluding 'providers' & 'text')
    ref text: https://docs.edenai.co/reference/text_generation_create
    ref image: https://docs.edenai.co/reference/text_generation_create
    """

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """extra parameters"""

    stop_sequences: Optional[List[str]] = None
    """Stop sequences to use."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["edenai_api_key"] = get_from_dict_or_env(
            values, "edenai_api_key", "EDENAI_API_KEY"
        )
        return values

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                logger.warning(
                    f"""{field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of model."""
        return "edenai"

    def _format_output(self, output: dict) -> str:
        if self.feature == "text":
            return output[self.provider]["generated_text"]
        else:
            return output[self.provider]["items"][0]["image"]

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to EdenAI's text generation endpoint.

        Args:
            prompt: The prompt to pass into the model.

        Returns:
            json formatted str response.

        """
        stops = None
        if self.stop_sequences is not None and stop is not None:
            raise ValueError(
                "stop sequences found in both the input and default params."
            )
        elif self.stop_sequences is not None:
            stops = self.stop_sequences
        else:
            stops = stop

        url = f"{self.base_url}/{self.feature}/{self.subfeature}"
        headers = {"Authorization": f"Bearer {self.edenai_api_key}"}
        payload = {
            **self.params,
            "providers": self.provider,
            "num_images": 1,  # always limit to 1 (ignored for text)
            "text": prompt,
            **kwargs,
        }
        request = Requests(headers=headers)

        response = request.post(url=url, data=payload)

        if response.status_code >= 500:
            raise Exception(f"EdenAI Server: Error {response.status_code}")
        elif response.status_code >= 400:
            raise ValueError(f"EdenAI received an invalid payload: {response.text}")
        elif response.status_code != 200:
            raise Exception(
                f"EdenAI returned an unexpected response with status "
                f"{response.status_code}: {response.text}"
            )

        output = self._format_output(response.json())

        if stops is not None:
            output = enforce_stop_tokens(output, stops)

        return output

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call EdenAi model to get predictions based on the prompt.

        Args:
            prompt: The prompt to pass into the model.
            stop: A list of stop words (optional).
            run_manager: A callback manager for async interaction with LLMs.

        Returns:
            The string generated by the model.
        """

        stops = None
        if self.stop_sequences is not None and stop is not None:
            raise ValueError(
                "stop sequences found in both the input and default params."
            )
        elif self.stop_sequences is not None:
            stops = self.stop_sequences
        else:
            stops = stop

        print("Running the acall")
        url = f"{self.base_url}/{self.feature}/{self.subfeature}"
        headers = {"Authorization": f"Bearer {self.edenai_api_key}"}
        payload = {
            **self.params,
            "providers": self.provider,
            "num_images": 1,  # always limit to 1 (ignored for text)
            "text": prompt,
            **kwargs,
        }

        async with ClientSession() as session:
            print("Requesting")
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status >= 500:
                    raise Exception(f"EdenAI Server: Error {response.status}")
                elif response.status >= 400:
                    raise ValueError(
                        f"EdenAI received an invalid payload: {response.text}"
                    )
                elif response.status != 200:
                    raise Exception(
                        f"EdenAI returned an unexpected response with status "
                        f"{response.status}: {response.text}"
                    )

                response_json = await response.json()

                output = self._format_output(response_json)
                if stops is not None:
                    output = enforce_stop_tokens(output, stops)

                return output
