import json
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import Field, SecretStr
from requests import Response


class SambaStudio(LLM):
    """
    SambaStudio large language models.

    Setup:
        To use, you should have the environment variables
        ``SAMBASTUDIO_URL`` set with your SambaStudio environment URL.
        ``SAMBASTUDIO_API_KEY``  set with your SambaStudio endpoint API key.
        https://sambanova.ai/products/enterprise-ai-platform-sambanova-suite
        read extra documentation in https://docs.sambanova.ai/sambastudio/latest/index.html
        Example:
        .. code-block:: python
            from langchain_community.llms.sambanova  import SambaStudio
            SambaStudio(
                sambastudio_url="your-SambaStudio-environment-URL",
                sambastudio_api_key="your-SambaStudio-API-key,
                model_kwargs={
                    "model" : model or expert name (set for Bundle endpoints),
                    "max_tokens" : max number of tokens to generate,
                    "temperature" : model temperature,
                    "top_p" : model top p,
                    "top_k" : model top k,
                    "do_sample" : wether to do sample
                    "process_prompt": wether to process prompt
                        (set for Bundle generic v1 and v2 endpoints)
                },
            )
    Key init args — completion params:
        model: str
            The name of the model to use, e.g., Meta-Llama-3-70B-Instruct-4096
            (set for Bundle endpoints).
        streaming: bool
            Whether to use streaming handler when using non streaming methods
        model_kwargs: dict
            Extra Key word arguments to pass to the model:
                max_tokens: int
                    max tokens to generate
                temperature: float
                    model temperature
                top_p: float
                    model top p
                top_k: int
                    model top k
                do_sample: bool
                    wether to do sample
                process_prompt:
                    wether to process prompt
                    (set for Bundle generic v1 and v2 endpoints)
    Key init args — client params:
        sambastudio_url: str
            SambaStudio endpoint Url
        sambastudio_api_key: str
            SambaStudio endpoint api key

    Instantiate:
        .. code-block:: python

            from langchain_community.llms import SambaStudio

            llm = SambaStudio=(
                sambastudio_url = set with your SambaStudio deployed endpoint URL,
                sambastudio_api_key = set with your SambaStudio deployed endpoint Key,
                model_kwargs = {
                    "model" : model or expert name (set for Bundle endpoints),
                    "max_tokens" : max number of tokens to generate,
                    "temperature" : model temperature,
                    "top_p" : model top p,
                    "top_k" : model top k,
                    "do_sample" : wether to do sample
                    "process_prompt" : wether to process prompt
                        (set for Bundle generic v1 and v2 endpoints)
                }
            )

    Invoke:
        .. code-block:: python
            prompt = "tell me a joke"
            response = llm.invoke(prompt)

    Stream:
        .. code-block:: python

        for chunk in llm.stream(prompt):
            print(chunk, end="", flush=True)

    Async:
        .. code-block:: python

        response = llm.ainvoke(prompt)
        await response

    """

    sambastudio_url: str = Field(default="")
    """SambaStudio Url"""

    sambastudio_api_key: SecretStr = Field(default=SecretStr(""))
    """SambaStudio api key"""

    base_url: str = Field(default="", exclude=True)
    """SambaStudio non streaming URL"""

    streaming_url: str = Field(default="", exclude=True)
    """SambaStudio streaming URL"""

    streaming: bool = Field(default=False)
    """Whether to use streaming handler when using non streaming methods"""

    model_kwargs: Optional[Dict[str, Any]] = None
    """Key word arguments to pass to the model."""

    class Config:
        populate_by_name = True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "sambastudio_url": "sambastudio_url",
            "sambastudio_api_key": "sambastudio_api_key",
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {"streaming": self.streaming, **{"model_kwargs": self.model_kwargs}}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "sambastudio-llm"

    def __init__(self, **kwargs: Any) -> None:
        """init and validate environment variables"""
        kwargs["sambastudio_url"] = get_from_dict_or_env(
            kwargs, "sambastudio_url", "SAMBASTUDIO_URL"
        )

        kwargs["sambastudio_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(kwargs, "sambastudio_api_key", "SAMBASTUDIO_API_KEY")
        )
        kwargs["base_url"], kwargs["streaming_url"] = self._get_sambastudio_urls(
            kwargs["sambastudio_url"]
        )
        super().__init__(**kwargs)

    def _get_sambastudio_urls(self, url: str) -> Tuple[str, str]:
        """
        Get streaming and non streaming URLs from the given URL

        Args:
            url: string with sambastudio base or streaming endpoint url

        Returns:
            base_url: string with url to do non streaming calls
            streaming_url: string with url to do streaming calls
        """
        if "openai" in url:
            base_url = url
            stream_url = url
        else:
            if "stream" in url:
                base_url = url.replace("stream/", "")
                stream_url = url
            else:
                base_url = url
                if "generic" in url:
                    stream_url = "generic/stream".join(url.split("generic"))
                else:
                    raise ValueError("Unsupported URL")
        return base_url, stream_url

    def _get_tuning_params(self, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get the tuning parameters to use when calling the LLM.

        Args:
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.

        Returns:
            The tuning parameters in the format required by api to use
        """
        if stop is None:
            stop = []

        # get the parameters to use when calling the LLM.
        _model_kwargs = self.model_kwargs or {}

        # handle the case where stop sequences are send in the invocation
        # and stop sequences has been also set in the model parameters
        _stop_sequences = _model_kwargs.get("stop_sequences", []) + stop
        if len(_stop_sequences) > 0:
            _model_kwargs["stop_sequences"] = _stop_sequences

        # set the parameters structure depending of the API
        if "openai" in self.sambastudio_url:
            if "select_expert" in _model_kwargs.keys():
                _model_kwargs["model"] = _model_kwargs.pop("select_expert")
            if "max_tokens_to_generate" in _model_kwargs.keys():
                _model_kwargs["max_tokens"] = _model_kwargs.pop(
                    "max_tokens_to_generate"
                )
            if "process_prompt" in _model_kwargs.keys():
                _model_kwargs.pop("process_prompt")
            tuning_params = _model_kwargs

        elif "api/v2/predict/generic" in self.sambastudio_url:
            if "model" in _model_kwargs.keys():
                _model_kwargs["select_expert"] = _model_kwargs.pop("model")
            if "max_tokens" in _model_kwargs.keys():
                _model_kwargs["max_tokens_to_generate"] = _model_kwargs.pop(
                    "max_tokens"
                )
            tuning_params = _model_kwargs

        elif "api/predict/generic" in self.sambastudio_url:
            if "model" in _model_kwargs.keys():
                _model_kwargs["select_expert"] = _model_kwargs.pop("model")
            if "max_tokens" in _model_kwargs.keys():
                _model_kwargs["max_tokens_to_generate"] = _model_kwargs.pop(
                    "max_tokens"
                )

            tuning_params = {
                k: {"type": type(v).__name__, "value": str(v)}
                for k, v in (_model_kwargs.items())
            }

        else:
            raise ValueError(
                f"Unsupported URL{self.sambastudio_url}"
                "only openai, generic v1 and generic v2 APIs are supported"
            )

        return tuning_params

    def _handle_request(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        streaming: Optional[bool] = False,
    ) -> Response:
        """
        Performs a post request to the LLM API.

        Args:
        prompt: The prompt to pass into the model
        stop: list of stop tokens
        streaming: wether to do a streaming call

        Returns:
            A request Response object
        """

        if isinstance(prompt, str):
            prompt = [prompt]

        params = self._get_tuning_params(stop)

        # create request payload for openAI v1 API
        if "openai" in self.sambastudio_url:
            messages_dict = [{"role": "user", "content": prompt[0]}]
            data = {"messages": messages_dict, "stream": streaming, **params}
            data = {key: value for key, value in data.items() if value is not None}
            headers = {
                "Authorization": f"Bearer "
                f"{self.sambastudio_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            }

        # create request payload for generic v1 API
        elif "api/v2/predict/generic" in self.sambastudio_url:
            if params.get("process_prompt", False):
                prompt = json.dumps(
                    {
                        "conversation_id": "sambaverse-conversation-id",
                        "messages": [
                            {"message_id": None, "role": "user", "content": prompt[0]}
                        ],
                    }
                )
            else:
                prompt = prompt[0]
            items = [{"id": "item0", "value": prompt}]
            params = {key: value for key, value in params.items() if value is not None}
            data = {"items": items, "params": params}
            headers = {"key": self.sambastudio_api_key.get_secret_value()}

        # create request payload for generic v1 API
        elif "api/predict/generic" in self.sambastudio_url:
            if params.get("process_prompt", False):
                if params["process_prompt"].get("value") == "True":
                    prompt = json.dumps(
                        {
                            "conversation_id": "sambaverse-conversation-id",
                            "messages": [
                                {
                                    "message_id": None,
                                    "role": "user",
                                    "content": prompt[0],
                                }
                            ],
                        }
                    )
                else:
                    prompt = prompt[0]
            else:
                prompt = prompt[0]
            if streaming:
                data = {"instance": prompt, "params": params}
            else:
                data = {"instances": [prompt], "params": params}
            headers = {"key": self.sambastudio_api_key.get_secret_value()}

        else:
            raise ValueError(
                f"Unsupported URL{self.sambastudio_url}"
                "only openai, generic v1 and generic v2 APIs are supported"
            )

        # make the request to SambaStudio API
        http_session = requests.Session()
        if streaming:
            response = http_session.post(
                self.streaming_url, headers=headers, json=data, stream=True
            )
        else:
            response = http_session.post(
                self.base_url, headers=headers, json=data, stream=False
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"Sambanova / complete call failed with status code "
                f"{response.status_code}."
                f"{response.text}."
            )
        return response

    def _process_response(self, response: Response) -> str:
        """
        Process a non streaming response from the api

        Args:
            response: A request Response object

        Returns
            completion: a string with model generation
        """

        # Extract json payload form response
        try:
            response_dict = response.json()
        except Exception as e:
            raise RuntimeError(
                f"Sambanova /complete call failed couldn't get JSON response {e}"
                f"response: {response.text}"
            )

        # process response payload for openai compatible API
        if "openai" in self.sambastudio_url:
            completion = response_dict["choices"][0]["message"]["content"]
        # process response payload for generic v2 API
        elif "api/v2/predict/generic" in self.sambastudio_url:
            completion = response_dict["items"][0]["value"]["completion"]
        # process response payload for generic v1 API
        elif "api/predict/generic" in self.sambastudio_url:
            completion = response_dict["predictions"][0]["completion"]
        else:
            raise ValueError(
                f"Unsupported URL{self.sambastudio_url}"
                "only openai, generic v1 and generic v2 APIs are supported"
            )
        return completion

    def _process_stream_response(self, response: Response) -> Iterator[GenerationChunk]:
        """
        Process a streaming response from the api

        Args:
            response: An iterable request Response object

        Yields:
            GenerationChunk: a GenerationChunk with model partial generation
        """

        try:
            import sseclient
        except ImportError:
            raise ImportError(
                "could not import sseclient library"
                "Please install it with `pip install sseclient-py`."
            )

        # process response payload for openai compatible API
        if "openai" in self.sambastudio_url:
            client = sseclient.SSEClient(response)
            for event in client.events():
                if event.event == "error_event":
                    raise RuntimeError(
                        f"Sambanova /complete call failed with status code "
                        f"{response.status_code}."
                        f"{event.data}."
                    )
                try:
                    # check if the response is not a final event ("[DONE]")
                    if event.data != "[DONE]":
                        if isinstance(event.data, str):
                            data = json.loads(event.data)
                        else:
                            raise RuntimeError(
                                f"Sambanova /complete call failed with status code "
                                f"{response.status_code}."
                                f"{event.data}."
                            )
                        if data.get("error"):
                            raise RuntimeError(
                                f"Sambanova /complete call failed with status code "
                                f"{response.status_code}."
                                f"{event.data}."
                            )
                        if len(data["choices"]) > 0:
                            content = data["choices"][0]["delta"]["content"]
                        else:
                            content = ""
                        generated_chunk = GenerationChunk(text=content)
                        yield generated_chunk

                except Exception as e:
                    raise RuntimeError(
                        f"Error getting content chunk raw streamed response: {e}"
                        f"data: {event.data}"
                    )

        # process response payload for generic v2 API
        elif "api/v2/predict/generic" in self.sambastudio_url:
            for line in response.iter_lines():
                try:
                    data = json.loads(line)
                    content = data["result"]["items"][0]["value"]["stream_token"]
                    generated_chunk = GenerationChunk(text=content)
                    yield generated_chunk

                except Exception as e:
                    raise RuntimeError(
                        f"Error getting content chunk raw streamed response: {e}"
                        f"line: {line}"
                    )

        # process response payload for generic v1 API
        elif "api/predict/generic" in self.sambastudio_url:
            for line in response.iter_lines():
                try:
                    data = json.loads(line)
                    content = data["result"]["responses"][0]["stream_token"]
                    generated_chunk = GenerationChunk(text=content)
                    yield generated_chunk

                except Exception as e:
                    raise RuntimeError(
                        f"Error getting content chunk raw streamed response: {e}"
                        f"line: {line}"
                    )

        else:
            raise ValueError(
                f"Unsupported URL{self.sambastudio_url}"
                "only openai, generic v1 and generic v2 APIs are supported"
            )

    def _stream(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Call out to Sambanova's complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: a list of strings on which the model should stop generating.
            run_manager: A run manager with callbacks for the LLM.
        Yields:
            chunk: GenerationChunk with model partial generation
        """
        response = self._handle_request(prompt, stop, streaming=True)
        for chunk in self._process_stream_response(response):
            if run_manager:
                run_manager.on_llm_new_token(chunk.text)
            yield chunk

    def _call(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Sambanova's complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: a list of strings on which the model should stop generating.

        Returns:
            result: string with model generation
        """
        if self.streaming:
            completion = ""
            for chunk in self._stream(
                prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
            ):
                completion += chunk.text

            return completion

        response = self._handle_request(prompt, stop, streaming=False)
        completion = self._process_response(response)
        return completion


class SambaNovaCloud(LLM):
    """
    SambaNova Cloud large language models.

    Setup:
        To use, you should have the environment variables:
        ``SAMBANOVA_URL`` set with SambaNova Cloud URL.
        defaults to http://cloud.sambanova.ai/
        ``SAMBANOVA_API_KEY`` set with your SambaNova Cloud API Key.
        Example:
        .. code-block:: python
            from langchain_community.llms.sambanova  import SambaNovaCloud
            SambaNovaCloud(
                sambanova_api_key="your-SambaNovaCloud-API-key,
                model = model name,
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                top_p = model top p,
                top_k = model top k
            )
    Key init args — completion params:
        model: str
            The name of the model to use, e.g., Meta-Llama-3-70B-Instruct-4096
            (set for CoE endpoints).
        streaming: bool
            Whether to use streaming handler when using non streaming methods
        max_tokens: int
            max tokens to generate
        temperature: float
            model temperature
        top_p: float
            model top p
        top_k: int
            model top k

    Key init args — client params:
        sambanova_url: str
            SambaNovaCloud Url defaults to http://cloud.sambanova.ai/
        sambanova_api_key: str
            SambaNovaCloud api key
    Instantiate:
        .. code-block:: python
            from langchain_community.llms.sambanova  import SambaNovaCloud
            SambaNovaCloud(
                sambanova_api_key="your-SambaNovaCloud-API-key,
                model = model name,
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                top_p = model top p,
                top_k = model top k
            )
    Invoke:
        .. code-block:: python
            prompt = "tell me a joke"
            response = llm.invoke(prompt)
    Stream:
        .. code-block:: python
        for chunk in llm.stream(prompt):
            print(chunk, end="", flush=True)
    Async:
        .. code-block:: python
        response = llm.ainvoke(prompt)
        await response
    """

    sambanova_url: str = Field(default="")
    """SambaNova Cloud Url"""

    sambanova_api_key: SecretStr = Field(default=SecretStr(""))
    """SambaNova Cloud api key"""

    model: str = Field(default="Meta-Llama-3.1-8B-Instruct")
    """The name of the model"""

    streaming: bool = Field(default=False)
    """Whether to use streaming handler when using non streaming methods"""

    max_tokens: int = Field(default=1024)
    """max tokens to generate"""

    temperature: float = Field(default=0.7)
    """model temperature"""

    top_p: Optional[float] = Field(default=None)
    """model top p"""

    top_k: Optional[int] = Field(default=None)
    """model top k"""

    stream_options: dict = Field(default={"include_usage": True})
    """stream options, include usage to get generation metrics"""

    class Config:
        populate_by_name = True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"sambanova_api_key": "sambanova_api_key"}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            "model": self.model,
            "streaming": self.streaming,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stream_options": self.stream_options,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "sambanovacloud-llm"

    def __init__(self, **kwargs: Any) -> None:
        """init and validate environment variables"""
        kwargs["sambanova_url"] = get_from_dict_or_env(
            kwargs,
            "sambanova_url",
            "SAMBANOVA_URL",
            default="https://api.sambanova.ai/v1/chat/completions",
        )
        kwargs["sambanova_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(kwargs, "sambanova_api_key", "SAMBANOVA_API_KEY")
        )
        super().__init__(**kwargs)

    def _handle_request(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        streaming: Optional[bool] = False,
    ) -> Response:
        """
        Performs a post request to the LLM API.

        Args:
            prompt: The prompt to pass into the model.
            stop: list of stop tokens

        Returns:
            A request Response object
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        messages_dict = [{"role": "user", "content": prompt[0]}]
        data = {
            "messages": messages_dict,
            "stream": streaming,
            "max_tokens": self.max_tokens,
            "stop": stop,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        data = {key: value for key, value in data.items() if value is not None}
        headers = {
            "Authorization": f"Bearer " f"{self.sambanova_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        http_session = requests.Session()
        if streaming:
            response = http_session.post(
                self.sambanova_url, headers=headers, json=data, stream=True
            )
        else:
            response = http_session.post(
                self.sambanova_url, headers=headers, json=data, stream=False
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Sambanova / complete call failed with status code "
                f"{response.status_code}."
                f"{response.text}."
            )
        return response

    def _process_response(self, response: Response) -> str:
        """
        Process a non streaming response from the api

        Args:
            response: A request Response object

        Returns
            completion: a string with model generation
        """

        # Extract json payload form response
        try:
            response_dict = response.json()
        except Exception as e:
            raise RuntimeError(
                f"Sambanova /complete call failed couldn't get JSON response {e}"
                f"response: {response.text}"
            )

        completion = response_dict["choices"][0]["message"]["content"]

        return completion

    def _process_stream_response(self, response: Response) -> Iterator[GenerationChunk]:
        """
        Process a streaming response from the api

        Args:
            response: An iterable request Response object

        Yields:
            GenerationChunk: a GenerationChunk with model partial generation
        """

        try:
            import sseclient
        except ImportError:
            raise ImportError(
                "could not import sseclient library"
                "Please install it with `pip install sseclient-py`."
            )

        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.event == "error_event":
                raise RuntimeError(
                    f"Sambanova /complete call failed with status code "
                    f"{response.status_code}."
                    f"{event.data}."
                )
            try:
                # check if the response is not a final event ("[DONE]")
                if event.data != "[DONE]":
                    if isinstance(event.data, str):
                        data = json.loads(event.data)
                    else:
                        raise RuntimeError(
                            f"Sambanova /complete call failed with status code "
                            f"{response.status_code}."
                            f"{event.data}."
                        )
                    if data.get("error"):
                        raise RuntimeError(
                            f"Sambanova /complete call failed with status code "
                            f"{response.status_code}."
                            f"{event.data}."
                        )
                    if len(data["choices"]) > 0:
                        content = data["choices"][0]["delta"]["content"]
                    else:
                        content = ""
                    generated_chunk = GenerationChunk(text=content)
                    yield generated_chunk

            except Exception as e:
                raise RuntimeError(
                    f"Error getting content chunk raw streamed response: {e}"
                    f"data: {event.data}"
                )

    def _call(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to SambaNovaCloud complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        if self.streaming:
            completion = ""
            for chunk in self._stream(
                prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
            ):
                completion += chunk.text

            return completion

        response = self._handle_request(prompt, stop, streaming=False)
        completion = self._process_response(response)
        return completion

    def _stream(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Call out to SambaNovaCloud complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        response = self._handle_request(prompt, stop, streaming=True)
        for chunk in self._process_stream_response(response):
            if run_manager:
                run_manager.on_llm_new_token(chunk.text)
            yield chunk
