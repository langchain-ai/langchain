from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)

from langchain.llms.base import LLM, create_base_retry_decorator
from langchain.pydantic_v1 import Field, root_validator


def _create_retry_decorator(
    retries: int,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    import aiohttp
    import requests

    errors = [requests.exceptions.RequestException, aiohttp.ClientError]
    try:
        import httpx

        errors.append(httpx.RequestError)
    except ImportError:
        pass
    return create_base_retry_decorator(
        error_types=errors, max_retries=retries, run_manager=run_manager
    )


def completion_with_retry(
    llm: LLMOverREST,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    retry_decorator = _create_retry_decorator(llm.retries, run_manager)

    @retry_decorator
    def _completion_with_retry() -> Any:
        return llm.http_call(json_body=json_body or {})

    return _completion_with_retry()


def _default_json_body_creator(
    prompt: str,
    stop: Optional[List[str]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    json_body: Dict[str, Any] = {"prompt": prompt}
    if stop:
        json_body["stop"] = stop
    if model_kwargs:
        json_body.update(model_kwargs)
    return json_body


class LLMOverREST(LLM):
    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    use_httpx: bool = False  #: :meta private:

    api_endpoint: str
    """The REST API endpoint from where the LLM will be served."""
    method: str = "POST"
    """The HTTP Method to use."""
    headers: Optional[Dict[str, str]] = None
    """A key-value map for all the request headers."""
    retries: int = 0
    """Number of retries to make in case the REST call fails"""
    timeout: float = 300.0
    """Timeout in seconds for the call to go through"""
    proxies: Optional[Dict[str, str]] = None
    """Dictionary containing the protocol to the proxy server configuration."""
    ssl_verify: Union[bool, str] = True
    """Either a boolean, which determines whether the server certificate is verified, 
    or a string, which should a path to a PEM file containing the trusted CAs,
    which will be used to verify the server certificate. 
    Note: Setting this parameter to ``False`` makes it vulnerable to MitM attacks 
    and is not recommended to be used in production."""
    client_cert: Optional[Union[str, Tuple[str, str]]] = None
    """The path to a client certificate or a Tuple containing the paths to the client 
    certificate and unencrypted private key file to be used."""
    model_kwargs: Dict[str, Any] = {}
    """Set of model parameters to be sent as part of the JSON body of the request. 
    This dict will be json-ified and appended to the request body."""
    json_body_creator: Callable[
        [str, Optional[List[str]], Dict[str, Any]], Dict[str, Any]
    ] = _default_json_body_creator
    """A key in the JSON response that corresponds to the generated text from the LLM. 
    This is used to extract the generated text and apply the stop sequence on it. 
    If this is ``None``, the ``stop`` parameter in the LLM call will not be honored. 
    Note: currently, it does not support nested keys and expect the keys to be 
    available at the root level of the response JSON"""

    @root_validator
    def validate_client(cls, values: Dict) -> Dict:
        if not values.get("client"):
            try:
                import httpx

                values["client"] = httpx.Client
                values["use_httpx"] = True
            except ImportError:
                import requests

                values["client"] = requests.Session
                values["use_httpx"] = False

        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        json_body = self.json_body_creator(prompt, stop, self.model_kwargs)
        try:
            response_text = completion_with_retry(
                llm=self, run_manager=run_manager, json_body=json_body
            )
        except BaseException as e:
            raise ValueError(f"Error faced during HTTP call: {e}")
        return response_text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "method": self.method,
            "endpoint": self.api_endpoint,
            "model_kwargs": self.model_kwargs or {},
        }

    @property
    def _llm_type(self) -> str:
        return "llm_over_rest"

    def http_call(self, json_body: Dict[str, str]) -> str:
        resp = (
            self._call_via_httpx(json_body)
            if self.use_httpx
            else self._call_via_requests(json_body)
        )
        return resp

    def _call_via_requests(self, json_body: Dict[str, Any]) -> str:
        import requests

        with self.client() as session:  # type: requests.Session
            response: requests.Response = session.request(
                method=self.method,
                url=self.api_endpoint,
                headers=self.headers,
                json=json_body,
                timeout=self.timeout,
                proxies=self.proxies,
                verify=self.ssl_verify,
                cert=self.client_cert,
            )
            response.raise_for_status()
            return response.text

    def _call_via_httpx(self, json_body: Dict[str, Any]) -> str:
        import httpx

        with self.client(
            verify=self.ssl_verify, cert=self.client_cert, proxies=self.proxies
        ) as client:  # type: httpx.Client
            response: httpx.Response = client.request(
                method=self.method,
                url=self.api_endpoint,
                json=json_body,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.text
