from typing import Any, Dict, List, Mapping, Optional

from langchain.callbacks.manager import (
    Callbacks,
)
from langchain.llms.base import LLM
from langchain.pydantic_v1 import Field
from langchain.utils import get_from_env


def _get_api_key() -> str:
    return get_from_env("api_key", "WATSONX_API_KEY")


class WatsonxLLM(LLM):
    """
    BAM LLM connector class to langchain
    WIP: is lacking some functions
    """

    model_name: str = "tiiuae/falcon-40b"
    api_key: str = Field(default_factory=_get_api_key)
    decoding_method: str = "sample"
    temperature: float = 0.05
    top_p: float = 1
    top_k: int = 50
    min_new_tokens: int = 1
    max_new_tokens: int = 100
    api_endpoint: str = "https://workbench-api.res.ibm.com/v1"
    repetition_penalty: Optional[float] = None
    random_seed: Optional[int] = None
    stop_sequences: Optional[list[str]] = None
    truncate_input_tokens: Optional[int] = None

    @property
    def _llm_type(self) -> str:
        return "custom"

    def __call__(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        try:
            import genai
        except ImportError as e:
            raise ImportError(
                "Cannot import genai, please install with "
                "`pip install ibm-generative-ai`."
            ) from e

        creds = genai.credentials.Credentials(api_key=self.api_key)
        gen_params = genai.schemas.generate_params.GenerateParams(
            decoding_method=self.decoding_method,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_new_tokens=self.min_new_tokens,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=self.repetition_penalty,
            random_seed=self.random_seed,
            stop_sequences=self.stop_sequences,
            truncate_input_tokens=self.truncate_input_tokens,
        )
        model = genai.model.Model(
            model=self.model_name, params=gen_params, credentials=creds
        )
        out = model.generate(prompts=[prompt])
        return out[0].generated_text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "decoding_method": self.decoding_method,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_new_tokens": self.min_new_tokens,
            "max_new_tokens": self.max_new_tokens,
            "api_key": self.api_key,
            "repetition_penalty": self.repetition_penalty,
            "random_seed": self.random_seed,
            "stop_sequences": self.stop_sequences,
            "truncate_input_tokens": self.truncate_input_tokens,
        }
