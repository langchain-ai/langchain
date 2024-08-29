from __future__ import annotations

import json
from io import StringIO
from typing import Any, Dict, Iterator, List, Optional

import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.utils import get_pydantic_field_names


class Llamafile(LLM):
    """Llamafile lets you distribute and run large language models with a
    single file.

    To get started, see: https://github.com/Mozilla-Ocho/llamafile

    To use this class, you will need to first:

    1. Download a llamafile.
    2. Make the downloaded file executable: `chmod +x path/to/model.llamafile`
    3. Start the llamafile in server mode:

        `./path/to/model.llamafile --server --nobrowser`

    Example:
        .. code-block:: python

            from langchain_community.llms import Llamafile
            llm = Llamafile()
            llm.invoke("Tell me a joke.")
    """

    base_url: str = "http://localhost:8080"
    """Base url where the llamafile server is listening."""

    request_timeout: Optional[int] = None
    """Timeout for server requests"""

    streaming: bool = False
    """Allows receiving each predicted token in real-time instead of
    waiting for the completion to finish. To enable this, set to true."""

    # Generation options

    seed: int = -1
    """Random Number Generator (RNG) seed. A random seed is used if this is 
    less than zero. Default: -1"""

    temperature: float = 0.8
    """Temperature. Default: 0.8"""

    top_k: int = 40
    """Limit the next token selection to the K most probable tokens. 
    Default: 40."""

    top_p: float = 0.95
    """Limit the next token selection to a subset of tokens with a cumulative 
    probability above a threshold P. Default: 0.95."""

    min_p: float = 0.05
    """The minimum probability for a token to be considered, relative to 
    the probability of the most likely token. Default: 0.05."""

    n_predict: int = -1
    """Set the maximum number of tokens to predict when generating text. 
    Note: May exceed the set limit slightly if the last token is a partial 
    multibyte character. When 0, no tokens will be generated but the prompt 
    is evaluated into the cache. Default: -1 = infinity."""

    n_keep: int = 0
    """Specify the number of tokens from the prompt to retain when the 
    context size is exceeded and tokens need to be discarded. By default, 
    this value is set to 0 (meaning no tokens are kept). Use -1 to retain all 
    tokens from the prompt."""

    tfs_z: float = 1.0
    """Enable tail free sampling with parameter z. Default: 1.0 = disabled."""

    typical_p: float = 1.0
    """Enable locally typical sampling with parameter p. 
    Default: 1.0 = disabled."""

    repeat_penalty: float = 1.1
    """Control the repetition of token sequences in the generated text. 
    Default: 1.1"""

    repeat_last_n: int = 64
    """Last n tokens to consider for penalizing repetition. Default: 64, 
    0 = disabled, -1 = ctx-size."""

    penalize_nl: bool = True
    """Penalize newline tokens when applying the repeat penalty. 
    Default: true."""

    presence_penalty: float = 0.0
    """Repeat alpha presence penalty. Default: 0.0 = disabled."""

    frequency_penalty: float = 0.0
    """Repeat alpha frequency penalty. Default: 0.0 = disabled"""

    mirostat: int = 0
    """Enable Mirostat sampling, controlling perplexity during text 
    generation. 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0. 
    Default: disabled."""

    mirostat_tau: float = 5.0
    """Set the Mirostat target entropy, parameter tau. Default: 5.0."""

    mirostat_eta: float = 0.1
    """Set the Mirostat learning rate, parameter eta. Default: 0.1."""

    class Config:
        extra = "forbid"

    @property
    def _llm_type(self) -> str:
        return "llamafile"

    @property
    def _param_fieldnames(self) -> List[str]:
        # Return the list of fieldnames that will be passed as configurable
        # generation options to the llamafile server. Exclude 'builtin' fields
        # from the BaseLLM class like 'metadata' as well as fields that should
        # not be passed in requests (base_url, request_timeout).
        ignore_keys = [
            "base_url",
            "cache",
            "callback_manager",
            "callbacks",
            "metadata",
            "name",
            "request_timeout",
            "streaming",
            "tags",
            "verbose",
            "custom_get_token_ids",
        ]
        attrs = [
            k for k in get_pydantic_field_names(self.__class__) if k not in ignore_keys
        ]
        return attrs

    @property
    def _default_params(self) -> Dict[str, Any]:
        params = {}
        for fieldname in self._param_fieldnames:
            params[fieldname] = getattr(self, fieldname)
        return params

    def _get_parameters(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        params = self._default_params

        # Only update keys that are already present in the default config.
        # This way, we don't accidentally post unknown/unhandled key/values
        # in the request to the llamafile server
        for k, v in kwargs.items():
            if k in params:
                params[k] = v

        if stop is not None and len(stop) > 0:
            params["stop"] = stop

        if self.streaming:
            params["stream"] = True

        return params

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Request prompt completion from the llamafile server and return the
        output.

        Args:
            prompt: The prompt to use for generation.
            stop: A list of strings to stop generation when encountered.
            run_manager:
            **kwargs: Any additional options to pass as part of the
            generation request.

        Returns:
            The string generated by the model.

        """

        if self.streaming:
            with StringIO() as buff:
                for chunk in self._stream(
                    prompt, stop=stop, run_manager=run_manager, **kwargs
                ):
                    buff.write(chunk.text)

                text = buff.getvalue()

            return text

        else:
            params = self._get_parameters(stop=stop, **kwargs)
            payload = {"prompt": prompt, **params}

            try:
                response = requests.post(
                    url=f"{self.base_url}/completion",
                    headers={
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    stream=False,
                    timeout=self.request_timeout,
                )
            except requests.exceptions.ConnectionError:
                raise requests.exceptions.ConnectionError(
                    f"Could not connect to Llamafile server. Please make sure "
                    f"that a server is running at {self.base_url}."
                )

            response.raise_for_status()
            response.encoding = "utf-8"

            text = response.json()["content"]

            return text

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Yields results objects as they are generated in real time.

        It also calls the callback manager's on_llm_new_token event with
        similar parameters to the OpenAI LLM class method of the same name.

        Args:
            prompt: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager:
            **kwargs: Any additional options to pass as part of the
            generation request.

        Returns:
            A generator representing the stream of tokens being generated.

        Yields:
            Dictionary-like objects each containing a token

        Example:
        .. code-block:: python

            from langchain_community.llms import Llamafile
            llm = Llamafile(
                temperature = 0.0
            )
            for chunk in llm.stream("Ask 'Hi, how are you?' like a pirate:'",
                    stop=["'","\n"]):
                result = chunk["choices"][0]
                print(result["text"], end='', flush=True)

        """
        params = self._get_parameters(stop=stop, **kwargs)
        if "stream" not in params:
            params["stream"] = True

        payload = {"prompt": prompt, **params}

        try:
            response = requests.post(
                url=f"{self.base_url}/completion",
                headers={
                    "Content-Type": "application/json",
                },
                json=payload,
                stream=True,
                timeout=self.request_timeout,
            )
        except requests.exceptions.ConnectionError:
            raise requests.exceptions.ConnectionError(
                f"Could not connect to Llamafile server. Please make sure "
                f"that a server is running at {self.base_url}."
            )

        response.encoding = "utf8"

        for raw_chunk in response.iter_lines(decode_unicode=True):
            content = self._get_chunk_content(raw_chunk)
            chunk = GenerationChunk(text=content)

            if run_manager:
                run_manager.on_llm_new_token(token=chunk.text)
            yield chunk

    def _get_chunk_content(self, chunk: str) -> str:
        """When streaming is turned on, llamafile server returns lines like:

        'data: {"content":" They","multimodal":true,"slot_id":0,"stop":false}'

        Here, we convert this to a dict and return the value of the 'content'
        field
        """

        if chunk.startswith("data:"):
            cleaned = chunk.lstrip("data: ")
            data = json.loads(cleaned)
            return data["content"]
        else:
            return chunk
