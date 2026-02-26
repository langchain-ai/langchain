"""Pass input through a moderation endpoint."""

from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.utils import check_package_version, get_from_dict_or_env
from pydantic import Field, model_validator
from typing_extensions import override

from langchain_classic.chains.base import Chain


class OpenAIModerationChain(Chain):
    """Pass input through a moderation endpoint.

    To use, you should have the `openai` python package installed, and the
    environment variable `OPENAI_API_KEY` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        ```python
        from langchain_classic.chains import OpenAIModerationChain

        moderation = OpenAIModerationChain()
        ```
    """

    client: Any = None
    async_client: Any = None
    model_name: str | None = None
    """Moderation model name to use."""
    error: bool = False
    """Whether or not to error if bad content was found."""
    input_key: str = "input"
    output_key: str = "output"
    openai_api_key: str | None = None
    openai_organization: str | None = None
    openai_pre_1_0: bool = Field(default=False)

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values,
            "openai_api_key",
            "OPENAI_API_KEY",
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
                values["client"] = openai.Moderation  # type: ignore[attr-defined]
            else:
                values["client"] = openai.OpenAI(api_key=openai_api_key)
                values["async_client"] = openai.AsyncOpenAI(api_key=openai_api_key)

        except ImportError as e:
            msg = (
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
            raise ImportError(msg) from e
        return values

    @property
    def input_keys(self) -> list[str]:
        """Expect input key."""
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """Return output key."""
        return [self.output_key]

    def _moderate(self, text: str, results: Any) -> str:
        condition = results["flagged"] if self.openai_pre_1_0 else results.flagged
        if condition:
            error_str = "Text was found that violates OpenAI's content policy."
            if self.error:
                raise ValueError(error_str)
            return error_str
        return text

    @override
    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
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
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        if self.openai_pre_1_0:
            return await super()._acall(inputs, run_manager=run_manager)
        text = inputs[self.input_key]
        results = await self.async_client.moderations.create(input=text)
        output = self._moderate(text, results.results[0])
        return {self.output_key: output}
