import logging
from typing import Any, Dict, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.pydantic_v1 import Extra, Field, root_validator
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class Petals(LLM):
    """Petals Bloom models.

    To use, you should have the ``petals`` python package installed, and the
    environment variable ``HUGGINGFACE_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.llms import petals
            petals = Petals()

    """

    client: Any
    """The client to use for the API calls."""

    tokenizer: Any
    """The tokenizer to use for the API calls."""

    model_name: str = "bigscience/bloom-petals"
    """The model to use."""

    temperature: float = 0.7
    """What sampling temperature to use"""

    max_new_tokens: int = 256
    """The maximum number of new tokens to generate in the completion."""

    top_p: float = 0.9
    """The cumulative probability for top-p sampling."""

    top_k: Optional[int] = None
    """The number of highest probability vocabulary tokens
    to keep for top-k-filtering."""

    do_sample: bool = True
    """Whether or not to use sampling; use greedy decoding otherwise."""

    max_length: Optional[int] = None
    """The maximum length of the sequence to be generated."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call
    not explicitly specified."""

    huggingface_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic config."""

        extra = Extra.forbid

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
        huggingface_api_key = get_from_dict_or_env(
            values, "huggingface_api_key", "HUGGINGFACE_API_KEY"
        )
        try:
            from petals import AutoDistributedModelForCausalLM
            from transformers import AutoTokenizer

            model_name = values["model_name"]
            values["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
            values["client"] = AutoDistributedModelForCausalLM.from_pretrained(
                model_name
            )
            values["huggingface_api_key"] = huggingface_api_key

        except ImportError:
            raise ValueError(
                "Could not import transformers or petals python package."
                "Please install with `pip install -U transformers petals`."
            )
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Petals API."""
        normal_params = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "max_length": self.max_length,
        }
        return {**normal_params, **self.model_kwargs}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "petals"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Petals API."""
        params = self._default_params
        params = {**params, **kwargs}
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        outputs = self.client.generate(inputs, **params)
        text = self.tokenizer.decode(outputs[0])
        if stop is not None:
            # I believe this is required since the stop tokens
            # are not enforced by the model parameters
            text = enforce_stop_tokens(text, stop)
        return text
