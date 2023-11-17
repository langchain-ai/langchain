from typing import Any, Dict, List, Optional, Sequence

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.pydantic_v1 import Extra, root_validator
from langchain.utils import convert_to_secret_str, get_from_dict_or_env


class AlephAlpha(LLM):
    """Aleph Alpha large language models.

    To use, you should have the ``aleph_alpha_client`` python package installed, and the
    environment variable ``ALEPH_ALPHA_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Parameters are explained more in depth here:
    https://github.com/Aleph-Alpha/aleph-alpha-client/blob/c14b7dd2b4325c7da0d6a119f6e76385800e097b/aleph_alpha_client/completion.py#L10

    Example:
        .. code-block:: python

            from langchain.llms import AlephAlpha
            aleph_alpha = AlephAlpha(aleph_alpha_api_key="my-api-key")
    """

    client: Any  #: :meta private:
    model: Optional[str] = "luminous-base"
    """Model name to use."""

    maximum_tokens: int = 64
    """The maximum number of tokens to be generated."""

    temperature: float = 0.0
    """A non-negative float that tunes the degree of randomness in generation."""

    top_k: int = 0
    """Number of most likely tokens to consider at each step."""

    top_p: float = 0.0
    """Total probability mass of tokens to consider at each step."""

    presence_penalty: float = 0.0
    """Penalizes repeated tokens."""

    frequency_penalty: float = 0.0
    """Penalizes repeated tokens according to frequency."""

    repetition_penalties_include_prompt: Optional[bool] = False
    """Flag deciding whether presence penalty or frequency penalty are
    updated from the prompt."""

    use_multiplicative_presence_penalty: Optional[bool] = False
    """Flag deciding whether presence penalty is applied
    multiplicatively (True) or additively (False)."""

    penalty_bias: Optional[str] = None
    """Penalty bias for the completion."""

    penalty_exceptions: Optional[List[str]] = None
    """List of strings that may be generated without penalty,
    regardless of other penalty settings"""

    penalty_exceptions_include_stop_sequences: Optional[bool] = None
    """Should stop_sequences be included in penalty_exceptions."""

    best_of: Optional[int] = None
    """returns the one with the "best of" results
    (highest log probability per token)
    """

    n: int = 1
    """How many completions to generate for each prompt."""

    logit_bias: Optional[Dict[int, float]] = None
    """The logit bias allows to influence the likelihood of generating tokens."""

    log_probs: Optional[int] = None
    """Number of top log probabilities to be returned for each generated token."""

    tokens: Optional[bool] = False
    """return tokens of completion."""

    disable_optimizations: Optional[bool] = False

    minimum_tokens: Optional[int] = 0
    """Generate at least this number of tokens."""

    echo: bool = False
    """Echo the prompt in the completion."""

    use_multiplicative_frequency_penalty: bool = False

    sequence_penalty: float = 0.0

    sequence_penalty_min_length: int = 2

    use_multiplicative_sequence_penalty: bool = False

    completion_bias_inclusion: Optional[Sequence[str]] = None

    completion_bias_inclusion_first_token_only: bool = False

    completion_bias_exclusion: Optional[Sequence[str]] = None

    completion_bias_exclusion_first_token_only: bool = False
    """Only consider the first token for the completion_bias_exclusion."""

    contextual_control_threshold: Optional[float] = None
    """If set to None, attention control parameters only apply to those tokens that have
    explicitly been set in the request.
    If set to a non-None value, control parameters are also applied to similar tokens.
    """

    control_log_additive: Optional[bool] = True
    """True: apply control by adding the log(control_factor) to attention scores.
    False: (attention_scores - - attention_scores.min(-1)) * control_factor
    """

    repetition_penalties_include_completion: bool = True
    """Flag deciding whether presence penalty or frequency penalty
    are updated from the completion."""

    raw_completion: bool = False
    """Force the raw completion of the model to be returned."""

    stop_sequences: Optional[List[str]] = None
    """Stop sequences to use."""

    # Client params
    aleph_alpha_api_key: Optional[str] = None
    """API key for Aleph Alpha API."""
    host: str = "https://api.aleph-alpha.com"
    """The hostname of the API host. 
    The default one is "https://api.aleph-alpha.com")"""
    hosting: Optional[str] = None
    """Determines in which datacenters the request may be processed.
    You can either set the parameter to "aleph-alpha" or omit it (defaulting to None).
    Not setting this value, or setting it to None, gives us maximal 
    flexibility in processing your request in our
    own datacenters and on servers hosted with other providers. 
    Choose this option for maximal availability.
    Setting it to "aleph-alpha" allows us to only process the 
    request in our own datacenters.
    Choose this option for maximal data privacy."""
    request_timeout_seconds: int = 305
    """Client timeout that will be set for HTTP requests in the 
    `requests` library's API calls.
    Server will close all requests after 300 seconds with an internal server error."""
    total_retries: int = 8
    """The number of retries made in case requests fail with certain retryable 
    status codes. If the last
    retry fails a corresponding exception is raised. Note, that between retries
    an exponential backoff
    is applied, starting with 0.5 s after the first retry and doubling for
    each retry made. So with the
    default setting of 8 retries a total wait time of 63.5 s is added 
    between the retries."""
    nice: bool = False
    """Setting this to True, will signal to the API that you intend to be 
    nice to other users
    by de-prioritizing your request below concurrent ones."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["aleph_alpha_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "aleph_alpha_api_key", "ALEPH_ALPHA_API_KEY")
        )
        try:
            from aleph_alpha_client import Client

            values["client"] = Client(
                token=values["aleph_alpha_api_key"].get_secret_value(),
                host=values["host"],
                hosting=values["hosting"],
                request_timeout_seconds=values["request_timeout_seconds"],
                total_retries=values["total_retries"],
                nice=values["nice"],
            )
        except ImportError:
            raise ImportError(
                "Could not import aleph_alpha_client python package. "
                "Please install it with `pip install aleph_alpha_client`."
            )
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling the Aleph Alpha API."""
        return {
            "maximum_tokens": self.maximum_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "n": self.n,
            "repetition_penalties_include_prompt": self.repetition_penalties_include_prompt,  # noqa: E501
            "use_multiplicative_presence_penalty": self.use_multiplicative_presence_penalty,  # noqa: E501
            "penalty_bias": self.penalty_bias,
            "penalty_exceptions": self.penalty_exceptions,
            "penalty_exceptions_include_stop_sequences": self.penalty_exceptions_include_stop_sequences,  # noqa: E501
            "best_of": self.best_of,
            "logit_bias": self.logit_bias,
            "log_probs": self.log_probs,
            "tokens": self.tokens,
            "disable_optimizations": self.disable_optimizations,
            "minimum_tokens": self.minimum_tokens,
            "echo": self.echo,
            "use_multiplicative_frequency_penalty": self.use_multiplicative_frequency_penalty,  # noqa: E501
            "sequence_penalty": self.sequence_penalty,
            "sequence_penalty_min_length": self.sequence_penalty_min_length,
            "use_multiplicative_sequence_penalty": self.use_multiplicative_sequence_penalty,  # noqa: E501
            "completion_bias_inclusion": self.completion_bias_inclusion,
            "completion_bias_inclusion_first_token_only": self.completion_bias_inclusion_first_token_only,  # noqa: E501
            "completion_bias_exclusion": self.completion_bias_exclusion,
            "completion_bias_exclusion_first_token_only": self.completion_bias_exclusion_first_token_only,  # noqa: E501
            "contextual_control_threshold": self.contextual_control_threshold,
            "control_log_additive": self.control_log_additive,
            "repetition_penalties_include_completion": self.repetition_penalties_include_completion,  # noqa: E501
            "raw_completion": self.raw_completion,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "aleph_alpha"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Aleph Alpha's completion endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = aleph_alpha("Tell me a joke.")
        """
        from aleph_alpha_client import CompletionRequest, Prompt

        params = self._default_params
        if self.stop_sequences is not None and stop is not None:
            raise ValueError(
                "stop sequences found in both the input and default params."
            )
        elif self.stop_sequences is not None:
            params["stop_sequences"] = self.stop_sequences
        else:
            params["stop_sequences"] = stop
        params = {**params, **kwargs}
        request = CompletionRequest(prompt=Prompt.from_text(prompt), **params)
        response = self.client.complete(model=self.model, request=request)
        text = response.completions[0].completion
        # If stop tokens are provided, Aleph Alpha's endpoint returns them.
        # In order to make this consistent with other endpoints, we strip them.
        if stop is not None or self.stop_sequences is not None:
            text = enforce_stop_tokens(text, params["stop_sequences"])
        return text


if __name__ == "__main__":
    aa = AlephAlpha()

    print(aa("How are you?"))
