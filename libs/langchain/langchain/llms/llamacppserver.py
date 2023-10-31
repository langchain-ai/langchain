import json
from typing import Any, Dict, Iterator, List, Mapping, Optional

import requests

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import BaseLLM
from langchain.pydantic_v1 import Extra
from langchain.schema import LLMResult
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output import GenerationChunk


def _stream_response_to_generation_chunk(
    stream_response: str,
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    parsed_response = json.loads(stream_response)
    generation_info = parsed_response if parsed_response.get("stop") is True else None
    return GenerationChunk(
        text=parsed_response.get("content", ""), generation_info=generation_info
    )


class _LlamaCppCommon(BaseLanguageModel):
    base_url: str = "http://localhost:8080"
    """Base url the model is hosted under."""

    grammar: Optional[str]
    logit_bias: Optional[int]
    mirostat: Optional[int] = 0
    mirostat_tau: Optional[int] = 5
    mirostat_eta: Optional[float] = 0.1
    n_predict: Optional[int] = 400
    n_probs: Optional[int] = 0
    frequency_penalty: Optional[float] = 0
    presence_penalty: Optional[float] = 0
    repeat_last_n: Optional[int] = 256
    repeat_penalty: Optional[float] = 1.18
    temperature: Optional[float] = 0.7
    typical_p: Optional[int] = 1
    seed: Optional[int] = 248
    stop: Optional[List[str]]
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.5
    tfs_z: Optional[int] = 1

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Llama.cpp."""
        return {
            "temperature": self.temperature,
            "repeat_last_n": self.repeat_last_n,
            "repeat_penalty": self.repeat_penalty,
            "top_k": self.top_p,
            "top_p": self.top_k,
            "tfs_z": self.tfs_z,
            "typical_p": self.typical_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "mirostat": self.mirostat,
            "mirostat_tau": self.mirostat_tau,
            "mirostat_eta": self.mirostat_eta,
            "n_predict": self.n_predict,
            "n_probs": self.n_probs,
            "grammar": self.grammar,
            "seed": self.seed,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}

    def _create_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        extra_headers: Optional[dict] = {},
        **kwargs: Any,
    ) -> Iterator[str]:
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop
        elif stop is None:
            stop = []
        params = {**self._default_params, "stop": stop, **kwargs}
        response = requests.post(
            url=f"{self.base_url}/completion",
            headers={"Content-Type": "application/json", **extra_headers},
            json={"prompt": prompt, **params},
            stream=True,
        )
        response.encoding = "utf-8"
        if response.status_code != 200:
            optional_detail = response.text
            raise ValueError(
                f"Llama.cpp call failed with status code {response.status_code}."
                f" Details: {optional_detail}"
            )
        return response.iter_lines(decode_unicode=True)

    def _stream_with_aggregation(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> GenerationChunk:
        final_chunk: Optional[GenerationChunk] = None
        for stream_resp in self._create_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_generation_chunk(stream_resp)
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from Llama-cpp stream.")

        return final_chunk


class LlamaCppServer(BaseLLM, _LlamaCppCommon):
    """LlamaCppServer connects to a llama.cpp server endpoint.

    To use, run ./server -m <model> from llama-cpp.

    Example:
        .. code-block:: python

            from langchain.llms import LlamaCppServer
            llm = LlamaCppServer(base_url="http://localhost:8080")
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "llama-cpp-server"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to Llama.cpp's generate endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = llm("Tell me a joke.")
        """
        # TODO: add caching here.
        generations = []
        for prompt in prompts:
            final_chunk = super()._stream_with_aggregation(
                prompt,
                stop=stop,
                run_manager=run_manager,
                verbose=self.verbose,
                **kwargs,
            )
            generations.append([final_chunk])
        return LLMResult(generations=generations)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for stream_resp in self._create_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_generation_chunk(stream_resp)
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )

JSON_GNBF = '''root  ::= object
value ::= object | array | string | number | boolean | "null"

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}"

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]"

string  ::=
  "\\"" (
    [^"\\\\] |
    "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\\"" ws

number  ::= "-"? [0-9]+ ws
boolean ::= ("true" | "false") ws

ws ::= ([ \\t\\n] ws)?'''

JSON_ARRAY_GNBF = '''root   ::= arr
value  ::= object | array | string | number | ("true" | "false" | "null") ws

arr  ::=
  "[\\n" ws (
            value
    (",\\n" ws value)*
  )? "]"

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\\"" (
    [^"\\\\] |
    "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \\t\\n] ws)?'''
