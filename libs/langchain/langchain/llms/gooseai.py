import logging
from typing import Any, Dict, List, Mapping, Optional

from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class GooseAI(LLM):
    """GooseAI large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``GOOSEAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.llms import GooseAI
            gooseai = GooseAI(model_name="gpt-neo-20b")

    """

    client: Any

    model_name: str = "gpt-neo-20b"
    """Model name to use"""

    temperature: float = 0.7
    """What sampling temperature to use"""

    max_tokens: int = 256
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""

    top_p: float = 1
    """Total probability mass of tokens to consider at each step."""

    min_tokens: int = 1
    """The minimum number of tokens to generate in the completion."""

    frequency_penalty: float = 0
    """Penalizes repeated tokens according to frequency."""

    presence_penalty: float = 0
    """Penalizes repeated tokens."""

    n: int = 1
    """How many completions to generate for each prompt."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    logit_bias: Optional[Dict[str, float]] = Field(default_factory=dict)
    """Adjust the probability of specific tokens being generated."""

    gooseai_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic config."""

        extra = Extra.ignore

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
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        gooseai_api_key = get_from_dict_or_env(
            values, "gooseai_api_key", "GOOSEAI_API_KEY"
        )
        try:
            import openai

            openai.api_key = gooseai_api_key
            openai.api_base = "https://api.goose.ai/v1"
            values["client"] = openai.Completion
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling GooseAI API."""
        normal_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "min_tokens": self.min_tokens,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
            "logit_bias": self.logit_bias,
        }
        return {**normal_params, **self.model_kwargs}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "gooseai"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the GooseAI API."""
        params = self._default_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop

        params = {**params, **kwargs}

        response = self.client.create(engine=self.model_name, prompt=prompt, **params)
        text = response.choices[0].text
        return text
