"""Pass input through a moderation endpoint."""

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.utils import check_package_version, get_from_dict_or_env
from pydantic import Field, model_validator

from langchain.chains.base import Chain


class OpenAIModerationChain(Chain):
    """Pass input through a moderation endpoint.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.chains import OpenAIModerationChain
            moderation = OpenAIModerationChain()
    """

    client: Any = None  #: :meta private:
    async_client: Any = None  #: :meta private:
    model_name: Optional[str] = None
    """Moderation model name to use."""
    error: bool = False
    """Whether or not to error if bad content was found."""
    input_key: str = "input"  #: :meta private:
    output_key: str = "output"  #: :meta private:
    openai_api_key: Optional[str] = None
    openai_organization: Optional[str] = None
    openai_pre_1_0: bool = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        openai_organization = get_from_dict_or_env(
            values,
            "openai_organization",
            "OPENAI_ORGANIZATION",
            default="",
        )
        try:
            import openai

            openai.api_key = openai_api_key
            if openai_organization:
                openai.organization = openai_organization
            values["openai_pre_1_0"] = False
            try:
                check_package_version("openai", gte_version="1.0")
            except ValueError:
                values["openai_pre_1_0"] = True
            if values["openai_pre_1_0"]:
                values["client"] = openai.Moderation
            else:
                values["client"] = openai.OpenAI()
                values["async_client"] = openai.AsyncOpenAI()

        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        return values

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return [self.output_key]

    def _moderate(self, text: str, results: Any) -> str:
        if self.openai_pre_1_0:
            condition = results["flagged"]
        else:
            condition = results.flagged
        if condition:
            error_str = "Text was found that violates OpenAI's content policy."
            if self.error:
                raise ValueError(error_str)
            else:
                return error_str
        return text

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        text = inputs[self.input_key]
        if self.openai_pre_1_0:
            results = self.client.create(text)
            output = self._moderate(text, results["results"][0])
        else:
            results = self.client.moderations.create(input=text)
            output = self._moderate(text, results.results[0])
        return {self.output_key: output}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        if self.openai_pre_1_0:
            return await super()._acall(inputs, run_manager=run_manager)
        text = inputs[self.input_key]
        results = await self.async_client.moderations.create(input=text)
        output = self._moderate(text, results.results[0])
        return {self.output_key: output}
