from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import Field
from langchain_core.utils import pre_init


class Aphrodite(BaseLLM):
    """Aphrodite language model."""

    model: str = ""
    """The name or path of a HuggingFace Transformers model."""

    tensor_parallel_size: Optional[int] = 1
    """The number of GPUs to use for distributed execution with tensor parallelism."""

    trust_remote_code: Optional[bool] = False
    """Trust remote code (e.g., from HuggingFace) when downloading the model 
    and tokenizer."""

    n: int = 1
    """Number of output sequences to return for the given prompt."""

    best_of: Optional[int] = None
    """Number of output sequences that are generated from the prompt.
    From these `best_of` sequences, the top `n` sequences are returned.
    `best_of` must be >= `n`. This is treated as the beam width when
    `use_beam_search` is True. By default, `best_of` is set to `n`."""

    presence_penalty: float = 0.0
    """Float that penalizes new tokens based on whether they appear in the 
    generated text so far. Values > 0 encourage the model to generate new
    tokens, while values < 0 encourage the model to repeat tokens."""

    frequency_penalty: float = 0.0
    """Float that penalizes new tokens based on their frequency in the 
    generated text so far. Applied additively to the logits."""

    repetition_penalty: float = 1.0
    """Float that penalizes new tokens based on their frequency in the
    generated text so far. Applied multiplicatively to the logits."""

    temperature: float = 1.0
    """Float that controls the randomness of the sampling. Lower values
    make the model more deterministic, while higher values make the model
    more random. Zero is equivalent to greedy sampling."""

    top_p: float = 1.0
    """Float that controls the cumulative probability of the top tokens to consider.
    Must be in (0, 1]. Set to 1.0 to consider all tokens."""

    top_k: int = -1
    """Integer that controls the number of top tokens to consider. Set to -1 to
    consider all tokens (disabled)."""

    top_a: float = 0.0
    """Float that controls the cutoff for Top-A sampling. Exact cutoff is
    top_a*max_prob**2. Must be in [0,inf], 0 to disable."""

    min_p: float = 0.0
    """Float that controls the cutoff for min-p sampling. Exact cutoff is
    min_p*max_prob. Must be in [0,1], 0 to disable."""

    tfs: float = 1.0
    """Float that controls the cumulative approximate curvature of the
    distribution to retain for Tail Free Sampling. Must be in (0, 1].
    Set to 1.0 to disable."""

    eta_cutoff: float = 0.0
    """Float that controls the cutoff threshold for Eta sampling
    (a form of entropy adaptive truncation sampling). Threshold is
    calculated as `min(eta, sqrt(eta)*entropy(probs)). Specified
    in units of 1e-4. Set to 0 to disable."""

    epsilon_cutoff: float = 0.0
    """Float that controls the cutoff threshold for Epsilon sampling
    (simple probability threshold truncation). Specified in units of
    1e-4. Set to 0 to disable."""

    typical_p: float = 1.0
    """Float that controls the cumulative probability of tokens closest
    in surprise to the expected surprise to consider. Must be in (0, 1].
    Set to 1 to disable."""

    mirostat_mode: int = 0
    """The mirostat mode to use. 0 for no mirostat, 2 for mirostat v2.
    Mode 1 is not supported."""

    mirostat_tau: float = 0.0
    """The target 'surprisal' that mirostat works towards. Range [0, inf)."""

    use_beam_search: bool = False
    """Whether to use beam search instead of sampling."""

    length_penalty: float = 1.0
    """Float that penalizes sequences based on their length. Used only
    when `use_beam_search` is True."""

    early_stopping: bool = False
    """Controls the stopping condition for beam search. It accepts the
    following values: `True`, where the generation stops as soon as there
    are `best_of` complete candidates; `False`, where a heuristic is applied
    to the generation stops when it is very unlikely to find better candidates;
    `never`, where the beam search procedure only stops where there cannot be
    better candidates (canonical beam search algorithm)."""

    stop: Optional[List[str]] = None
    """List of strings that stop the generation when they are generated.
    The returned output will not contain the stop tokens."""

    stop_token_ids: Optional[List[int]] = None
    """List of tokens that stop the generation when they are generated.
    The returned output will contain the stop tokens unless the stop tokens
    are special tokens."""

    ignore_eos: bool = False
    """Whether to ignore the EOS token and continue generating tokens after 
    the EOS token is generated."""

    max_tokens: int = 512
    """Maximum number of tokens to generate per output sequence."""

    logprobs: Optional[int] = None
    """Number of log probabilities to return per output token."""

    prompt_logprobs: Optional[int] = None
    """Number of log probabilities to return per prompt token."""

    custom_token_bans: Optional[List[int]] = None
    """List of token IDs to ban from generating."""

    skip_special_tokens: bool = True
    """Whether to skip special tokens in the output. Defaults to True."""

    spaces_between_special_tokens: bool = True
    """Whether to add spaces between special tokens in the output.
    Defaults to True."""

    logit_bias: Optional[Dict[str, float]] = None
    """List of LogitsProcessors to change the probability of token
    prediction at runtime."""

    dtype: str = "auto"
    """The data type for the model weights and activations."""

    download_dir: Optional[str] = None
    """Directory to download and load the weights. (Default to the default 
    cache dir of huggingface)"""

    quantization: Optional[str] = None
    """Quantization mode to use. Can be one of `awq` or `gptq`."""

    aphrodite_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `aphrodite.LLM` call not explicitly
    specified."""

    client: Any  #: :meta private:

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

        try:
            from aphrodite import LLM as AphroditeModel
        except ImportError:
            raise ImportError(
                "Could not import aphrodite-engine python package. "
                "Please install it with `pip install aphrodite-engine`."
            )

        # aphrodite_kwargs = values["aphrodite_kwargs"]
        # if values.get("quantization"):
        #     aphrodite_kwargs["quantization"] = values["quantization"]

        values["client"] = AphroditeModel(
            model=values["model"],
            tensor_parallel_size=values["tensor_parallel_size"],
            trust_remote_code=values["trust_remote_code"],
            dtype=values["dtype"],
            download_dir=values["download_dir"],
            **values["aphrodite_kwargs"],
        )

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling aphrodite."""
        return {
            "n": self.n,
            "best_of": self.best_of,
            "max_tokens": self.max_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "top_a": self.top_a,
            "min_p": self.min_p,
            "temperature": self.temperature,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
            "tfs": self.tfs,
            "eta_cutoff": self.eta_cutoff,
            "epsilon_cutoff": self.epsilon_cutoff,
            "typical_p": self.typical_p,
            "mirostat_mode": self.mirostat_mode,
            "mirostat_tau": self.mirostat_tau,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
            "use_beam_search": self.use_beam_search,
            "stop": self.stop,
            "ignore_eos": self.ignore_eos,
            "logprobs": self.logprobs,
            "prompt_logprobs": self.prompt_logprobs,
            "custom_token_bans": self.custom_token_bans,
            "skip_special_tokens": self.skip_special_tokens,
            "spaces_between_special_tokens": self.spaces_between_special_tokens,
            "logit_bias": self.logit_bias,
        }

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""

        from aphrodite import SamplingParams

        # build sampling parameters
        params = {**self._default_params, **kwargs, "stop": stop}
        if "logit_bias" in params:
            del params["logit_bias"]
        sampling_params = SamplingParams(**params)
        # call the model
        outputs = self.client.generate(prompts, sampling_params)

        generations = []
        for output in outputs:
            text = output.outputs[0].text
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "aphrodite"
