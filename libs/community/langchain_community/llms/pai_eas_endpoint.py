import json
import logging
from typing import Any, Dict, Iterator, List, Mapping, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.utils import get_from_dict_or_env, pre_init

from langchain_community.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__name__)


class PaiEasEndpoint(LLM):
    """Langchain LLM class to help to access eass llm service.

        To use this endpoint, must have a deployed eas chat llm service on PAI AliCloud.
    One can set the environment variable ``eas_service_url`` and ``eas_service_token``.
    The environment variables can set with your eas service url and service token.

    Example:
        .. code-block:: python

            from langchain_community.llms.pai_eas_endpoint import PaiEasEndpoint
            eas_chat_endpoint = PaiEasChatEndpoint(
                eas_service_url="your_service_url",
                eas_service_token="your_service_token"
            )
    """

    """PAI-EAS Service URL"""
    eas_service_url: str

    """PAI-EAS Service TOKEN"""
    eas_service_token: str

    """PAI-EAS Service Infer Params"""
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.95
    top_p: Optional[float] = 0.1
    top_k: Optional[int] = 0
    stop_sequences: Optional[List[str]] = None

    """Enable stream chat mode."""
    streaming: bool = False

    """Key/value arguments to pass to the model. Reserved for future use"""
    model_kwargs: Optional[dict] = None

    version: Optional[str] = "2.0"

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["eas_service_url"] = get_from_dict_or_env(
            values, "eas_service_url", "EAS_SERVICE_URL"
        )
        values["eas_service_token"] = get_from_dict_or_env(
            values, "eas_service_token", "EAS_SERVICE_TOKEN"
        )

        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "pai_eas_endpoint"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Cohere API."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "stop_sequences": [],
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            "eas_service_url": self.eas_service_url,
            "eas_service_token": self.eas_service_token,
            **_model_kwargs,
        }

    def _invocation_params(
        self, stop_sequences: Optional[List[str]], **kwargs: Any
    ) -> dict:
        params = self._default_params
        if self.stop_sequences is not None and stop_sequences is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop_sequences is not None:
            params["stop"] = self.stop_sequences
        else:
            params["stop"] = stop_sequences
        if self.model_kwargs:
            params.update(self.model_kwargs)
        return {**params, **kwargs}

    @staticmethod
    def _process_response(
        response: Any, stop: Optional[List[str]], version: Optional[str]
    ) -> str:
        if version == "1.0":
            text = response
        else:
            text = response["response"]

        if stop:
            text = enforce_stop_tokens(text, stop)
        return "".join(text)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        params = self._invocation_params(stop, **kwargs)
        prompt = prompt.strip()
        response = None
        try:
            if self.streaming:
                completion = ""
                for chunk in self._stream(prompt, stop, run_manager, **params):
                    completion += chunk.text
                return completion
            else:
                response = self._call_eas(prompt, params)
                _stop = params.get("stop")
                return self._process_response(response, _stop, self.version)
        except Exception as error:
            raise ValueError(f"Error raised by the service: {error}")

    def _call_eas(self, prompt: str = "", params: Dict = {}) -> Any:
        """Generate text from the eas service."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"{self.eas_service_token}",
        }
        if self.version == "1.0":
            body = {
                "input_ids": f"{prompt}",
            }
        else:
            body = {
                "prompt": f"{prompt}",
            }

        # add params to body
        for key, value in params.items():
            body[key] = value

        # make request
        response = requests.post(self.eas_service_url, headers=headers, json=body)

        if response.status_code != 200:
            raise Exception(
                f"Request failed with status code {response.status_code}"
                f" and message {response.text}"
            )

        try:
            return json.loads(response.text)
        except Exception as e:
            if isinstance(e, json.decoder.JSONDecodeError):
                return response.text
            raise e

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        invocation_params = self._invocation_params(stop, **kwargs)

        headers = {
            "User-Agent": "Test Client",
            "Authorization": f"{self.eas_service_token}",
        }

        if self.version == "1.0":
            pload = {"input_ids": prompt, **invocation_params}
            response = requests.post(
                self.eas_service_url, headers=headers, json=pload, stream=True
            )

            res = GenerationChunk(text=response.text)

            if run_manager:
                run_manager.on_llm_new_token(res.text)

            # yield text, if any
            yield res
        else:
            pload = {"prompt": prompt, "use_stream_chat": "True", **invocation_params}

            response = requests.post(
                self.eas_service_url, headers=headers, json=pload, stream=True
            )

            for chunk in response.iter_lines(
                chunk_size=8192, decode_unicode=False, delimiter=b"\0"
            ):
                if chunk:
                    data = json.loads(chunk.decode("utf-8"))
                    output = data["response"]
                    # identify stop sequence in generated text, if any
                    stop_seq_found: Optional[str] = None
                    for stop_seq in invocation_params["stop"]:
                        if stop_seq in output:
                            stop_seq_found = stop_seq

                    # identify text to yield
                    text: Optional[str] = None
                    if stop_seq_found:
                        text = output[: output.index(stop_seq_found)]
                    else:
                        text = output

                    # yield text, if any
                    if text:
                        res = GenerationChunk(text=text)
                        if run_manager:
                            run_manager.on_llm_new_token(res.text)
                        yield res

                    # break if stop sequence found
                    if stop_seq_found:
                        break
