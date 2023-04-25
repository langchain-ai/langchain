"""Wrapper around oobabooga/text-generation-webui blocking API."""

# TODO: streaming API


from typing import Any, Dict, List, Mapping, Optional

import requests
from pydantic import BaseModel, Extra, root_validator

from langchain.llms.base import LLM


class TextGenerationWebui(LLM, BaseModel):
    """A client for text-generation-webui's blocking API.

    Example:
        .. code-block:: python

            from langchain.llms import TextGenerationWebui
            tgwui = TextGenerationWebui()
    """

    max_length : int = 200
    """The maximum length the generated tokens can have. Corresponds to the length of the input prompt + max_new_tokens. Its effect is overridden by max_new_tokens, if also set."""
    max_new_tokens : int = 200
    """The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."""
    do_sample : bool = True
    """Whether or not to use sampling ; use greedy decoding otherwise."""
    top_p : float = 1.0
    """If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation."""
    typical_p : float = 1.0
    """Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to typical_p or higher are kept for generation."""
    repetition_penalty : float = 1.1
    """The parameter for repetition penalty. 1.0 means no penalty."""
    encoder_repetition_penalty : float = 1.0
    """The parameter for encoder_repetition_penalty. An exponential penalty on sequences that are not in the original input. 1.0 means no penalty."""
    top_k : int = 0
    """The number of highest probability vocabulary tokens to keep for top-k-filtering."""
    min_length : int = 0
    """The minimum length of the sequence to be generated. Corresponds to the length of the input prompt + min_new_tokens. Its effect is overridden by min_new_tokens, if also set."""
    no_repeat_ngram_size : int = 0
    """If set to int > 0, all ngrams of that size can only occur once."""
    num_beams : int = 1
    """Number of beams for beam search. 1 means no beam search."""
    penalty_alpha : float = 0
    """The values balance the model confidence and the degeneration penalty in contrastive search decoding."""
    length_penalty : float = 1
    """Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences."""
    early_stopping : bool = False
    """Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values: True, where the generation stops as soon as there are num_beams complete candidates; False, where an heuristic is applied and the generation stops when is it very unlikely to find better candidates; "never", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm)."""
    seed : int = -1
    """Random seed to control sampling, containing two integers, used when do_sample is True. See the seed argument from stateless functions in tf.random. kwargs — Ad hoc parametrization of generate_config and/or additional model-specific kwargs that will be forwarded to the forward function of the model. If the model is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with decoder_."""
    add_bos_token : int = 1
    """Add the bos_token to the beginning of prompts. Disabling this can make the replies more creative. """
    truncation_length : int = 2048
    """The leftmost tokens are removed if the prompt exceeds this length. Most models require this to be at most 2048."""
    ban_eos_token : bool = False
    """Forces the model to never end the generation prematurely."""
    skip_special_tokens : bool = True
    """Skip special tokens."""
    custom_stopping_strings : str = ""
    """In addition to the defaults. Separated by commas. For instance: \"\nYour Assistant:\", \"\nThe assistant:\""""
    stopping_strings : list = []
    """List of strings to use as stopping strings."""

    url: str = "http://127.0.0.1:5000/api/v1"
    """Base url of the API."""

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling the API."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "top_p": self.top_p,
            "typical_p": self.typical_p,
            "repetition_penalty": self.repetition_penalty,
            "encoder_repetition_penalty": self.encoder_repetition_penalty,
            "top_k": self.top_k,
            "min_length": self.min_length,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "num_beams": self.num_beams,
            "penalty_alpha": self.penalty_alpha,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
            "seed": self.seed,
            "add_bos_token": self.add_bos_token,
            "truncation_length": self.truncation_length,
            "ban_eos_token": self.ban_eos_token,
            "skip_special_tokens": self.skip_special_tokens,
            "custom_stopping_strings": self.custom_stopping_strings,
            "stopping_strings": self.stopping_strings
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"url": self.url}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        response = requests.get(
            url=self.url + "/model",
        )
        response_json = response.json()
        return response_json["result"]

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call TextGenerationWebui's generate endpoint.

        Args:
            prompt: The prompt to pass to the API.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = TextGenerationWebui("Tell me a joke.")
        """
        response = requests.post(
            url=self.url + "/generate",
            headers={
                "Content-Type": "application/json",
            },
            json={"prompt": prompt,
                  "stopping_strings": stop,
                  **self._default_params}
        )
        response_json = response.json()
        text = response_json["results"][0]["text"]
        return text

    def get_num_tokens(self, text: str) -> int:
        """Call TextGenerationWebui's token-count endpoint.

        Args:
            prompt: The prompt to pass to the API.

        Returns:
            The number of tokens reported by the API.

        Example:
            .. code-block:: python

                response = TextGenerationWebui.get_num_tokens("Tell me a joke.")
        """
        response = requests.post(
            url=self.url + "/token-count",
            headers={
                "Content-Type": "application/json",
            },
            json={"prompt": text}
        )
        response_json = response.json()
        tokens = response_json["results"][0]["tokens"]
        return tokens
