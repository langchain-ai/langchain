import re
from typing import Any, Iterator, List, Mapping, Optional

import requests

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.schema.output import GenerationChunk


class TitanTakeoff(LLM):
    port: int = 8000
    """Specifies the port to use for the Titan Takeoff API. Default = 8000."""
    generate_max_length: int = 128
    sampling_topk: int = 1
    sampling_topp: float = 1.0
    sampling_temperature: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    streaming: bool = False

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling Titan Takeoff Server."""
        params = {}
        if self.generate_max_length:
            params["generate_max_length"] = self.generate_max_length
        if self.sampling_topk:
            params["sampling_topk"] = self.sampling_topk
        if self.sampling_topp:
            params["sampling_topp"] = self.sampling_topp
        if self.sampling_temperature:
            params["sampling_temperature"] = self.sampling_temperature
        if self.repetition_penalty:
            params["repetition_penalty"] = self.repetition_penalty
        if self.no_repeat_ngram_size:
            params["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        return params

    @property
    def _llm_type(self) -> str:
        return "titan_takeoff"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if self.streaming:
            text_output = ""
            for chunk in self._stream(
                prompt=prompt,
                stop=stop,
                run_manager=run_manager,
            ):
                text_output += chunk.text
            return text_output

        url = f"http://localhost:{self.port}/generate"
        params = {"text": prompt, **self._default_params}

        response = requests.post(url, json=params)
        response.encoding = "utf-8"
        text = ""

        if "message" in response.json():
            text = response.json()["message"]
        else:
            raise ValueError("Something went wrong.")
        if stop is None:
            text = enforce_stop_tokens(text, [re.escape("<|endoftext|>")])
        else:
            text = enforce_stop_tokens(text, stop)
        return text

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Iterator[GenerationChunk]:
        url = f"http://localhost:{self.port}/generate_stream"
        params = {"text": prompt, **self._default_params}

        response = requests.post(url, json=params, stream=True)
        response.encoding = "utf-8"
        for text in response.iter_content(chunk_size=1, decode_unicode=True):
            if text:
                chunk = GenerationChunk(text=text)
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(token=chunk.text)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"port": self.port, **{}, **self._default_params}
