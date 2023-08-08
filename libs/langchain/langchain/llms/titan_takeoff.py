import re
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

class TitanTakeoff(LLM):
    port: int = 8000
    """Specifies the port to use for the Titan Takeoff API. Default = 8000."""
    generate_max_length: int = 128
    sampling_topk: int = 1
    sampling_topp: float = 1.0
    sampling_temperature: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    @property
    def _llm_type(self) -> str:
        return "titan_takeoff"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]],
    ) -> str:
        import requests

        url = f"http://localhost:{self.port}/generate"
        json = {
            "text": prompt,
            "generate_max_length": self.generate_max_length,
            "sampling_topk": self.sampling_topk,
            "sampling_topp": self.sampling_topp,
            "sampling_temperature": self.sampling_temperature,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
        }

        response = requests.post(url, json=json)
        response.encoding = 'utf-8'
        text = ""

        if "message" in response.json():
            text = response.json()["message"]
        else:
            raise ValueError("Something went wrong.")
        if stop is None:
            text = enforce_stop_tokens(text, [re.escape('<|endoftext|>')])
        else:
            text = enforce_stop_tokens(text, stop)
        return text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"port": self.port}