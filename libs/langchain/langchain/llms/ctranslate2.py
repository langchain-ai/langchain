from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import BaseLLM
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema.output import Generation, LLMResult


class CTranslate2(BaseLLM):
    """CTranslate2 language model."""

    model_path: str = ""
    """Path to the CTranslate2 model directory."""

    tokenizer_name: str = ""
    """Name of the original Hugging Face model needed to load the proper tokenizer."""

    device: str = "cpu"
    """Device to use (possible values are: cpu, cuda, auto)."""

    device_index: Union[int, List[int]] = 0
    """Device IDs where to place this generator on."""

    compute_type: Union[str, Dict[str, str]] = "default"
    """
    Model computation type or a dictionary mapping a device name to the computation type
    (possible values are: default, auto, int8, int8_float32, int8_float16,
    int8_bfloat16, int16, float16, bfloat16, float32).
    """

    max_length: int = 512
    """Maximum generation length."""

    sampling_topk: int = 1
    """Randomly sample predictions from the top K candidates."""

    sampling_topp: float = 1
    """Keep the most probable tokens whose cumulative probability exceeds this value."""

    sampling_temperature: float = 1
    """Sampling temperature to generate more random samples."""

    client: Any  #: :meta private:

    tokenizer: Any  #: :meta private:

    ctranslate2_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """
    Holds any model parameters valid for `ctranslate2.Generator` call not 
    explicitly specified.
    """

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

        try:
            import ctranslate2
        except ImportError:
            raise ImportError(
                "Could not import ctranslate2 python package. "
                "Please install it with `pip install ctranslate2`."
            )

        try:
            import transformers
        except ImportError:
            raise ImportError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )

        values["client"] = ctranslate2.Generator(
            model_path=values["model_path"],
            device=values["device"],
            device_index=values["device_index"],
            compute_type=values["compute_type"],
            **values["ctranslate2_kwargs"],
        )

        values["tokenizer"] = transformers.AutoTokenizer.from_pretrained(
            values["tokenizer_name"]
        )

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters."""
        return {
            "max_length": self.max_length,
            "sampling_topk": self.sampling_topk,
            "sampling_topp": self.sampling_topp,
            "sampling_temperature": self.sampling_temperature,
        }

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        # build sampling parameters
        params = {**self._default_params, **kwargs}

        # call the model
        encoded_prompts = self.tokenizer(prompts)["input_ids"]
        tokenized_prompts = [
            self.tokenizer.convert_ids_to_tokens(encoded_prompt)
            for encoded_prompt in encoded_prompts
        ]

        results = self.client.generate_batch(tokenized_prompts, **params)

        sequences = [result.sequences_ids[0] for result in results]
        decoded_sequences = [self.tokenizer.decode(seq) for seq in sequences]

        generations = []
        for text in decoded_sequences:
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "ctranslate2"
