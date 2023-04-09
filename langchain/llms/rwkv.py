"""Wrapper for the RWKV model.

Based on https://github.com/saharNooby/rwkv.cpp/blob/master/rwkv/chat_with_bot.py
         https://github.com/BlinkDL/ChatRWKV/blob/main/v2/chat.py
"""
from typing import Any, Dict, List, Mapping, Optional, Set, SupportsIndex

from pydantic import BaseModel, Extra, root_validator

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens


class RWKV(LLM, BaseModel):
    r"""Wrapper around RWKV language models.

    To use, you should have the ``rwkv`` python package installed, the
    pre-trained model file, and the model's config information.

    Example:
        .. code-block:: python

            from langchain.llms import RWKV
            model = RWKV(model="./models/rwkv-3b-fp16.bin", strategy="cpu fp32")

            # Simplest invocation
            response = model("Once upon a time, ")
    """

    model: str
    """Path to the pre-trained RWKV model file."""

    tokens_path: str
    """Path to the RWKV tokens file."""

    strategy: str = "cpu fp32"
    """Token context window."""

    rwkv_verbose: bool = True
    """Print debug information."""

    temperature: float = 1.0
    """The temperature to use for sampling."""

    top_p: float = 0.5
    """The top-p value to use for sampling."""

    penalty_alpha_frequency: float = 0.4
    """Positive values penalize new tokens based on their existing frequency
    in the text so far, decreasing the model's likelihood to repeat the same
    line verbatim.."""

    penalty_alpha_presence: float = 0.4
    """Positive values penalize new tokens based on whether they appear
    in the text so far, increasing the model's likelihood to talk about
    new topics.."""

    CHUNK_LEN: int = 256
    """Batch size for prompt processing."""

    max_tokens_per_generation: SupportsIndex = 256
    """Maximum number of tokens to generate."""

    client: Any = None  #: :meta private:

    tokenizer: Any = None  #: :meta private:

    pipeline: Any = None  #: :meta private:

    model_tokens: Any = None  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "verbose": self.verbose,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "penalty_alpha_frequency": self.penalty_alpha_frequency,
            "penalty_alpha_presence": self.penalty_alpha_presence,
            "CHUNK_LEN": self.CHUNK_LEN,
            "max_tokens_per_generation": self.max_tokens_per_generation,
        }

    @staticmethod
    def _rwkv_param_names() -> Set[str]:
        """Get the identifying parameters."""
        return {
            "verbose",
        }

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in the environment."""
        try:
            import tokenizers
        except ImportError:
            raise ValueError(
                "Could not import tokenizers python package. "
                "Please install it with `pip install tokenizers`."
            )
        try:
            from rwkv.model import RWKV as RWKVMODEL
            from rwkv.utils import PIPELINE

            values["tokenizer"] = tokenizers.Tokenizer.from_file(values["tokens_path"])

            rwkv_keys = cls._rwkv_param_names()
            model_kwargs = {k: v for k, v in values.items() if k in rwkv_keys}
            model_kwargs["verbose"] = values["rwkv_verbose"]
            values["client"] = RWKVMODEL(
                values["model"], strategy=values["strategy"], **model_kwargs
            )
            values["pipeline"] = PIPELINE(values["client"], values["tokens_path"])

        except ImportError:
            raise ValueError(
                "Could not import rwkv python package. "
                "Please install it with `pip install rwkv`."
            )
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            **self._default_params,
            **{k: v for k, v in self.__dict__.items() if k in RWKV._rwkv_param_names()},
        }

    @property
    def _llm_type(self) -> str:
        """Return the type of llm."""
        return "rwkv-4"

    def rwkv_generate(self, prompt: str) -> str:
        tokens = self.tokenizer.encode(prompt).ids

        logits = None
        state = None

        occurrence = {}

        # Feed in the input string
        while len(tokens) > 0:
            logits, state = self.client.forward(tokens[: self.CHUNK_LEN], state)
            tokens = tokens[self.CHUNK_LEN :]

        decoded = ""
        for i in range(self.max_tokens_per_generation):
            token = self.pipeline.sample_logits(
                logits, temperature=self.temperature, top_p=self.top_p
            )

            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

            decoded += self.tokenizer.decode([token])

            if "\n" in decoded:
                break

            # feed back in
            logits, state = self.client.forward([token], state)
            for n in occurrence:
                logits[n] -= (
                    self.penalty_alpha_presence
                    + occurrence[n] * self.penalty_alpha_frequency
                )

        return decoded

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        r"""RWKV generation

        Args:
            prompt: The prompt to pass into the model.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                prompt = "Once upon a time, "
                response = model(prompt, n_predict=55)
        """
        text = self.rwkv_generate(prompt)

        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text
