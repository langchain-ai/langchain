from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens


def _get_generated_text(resp: Dict[str, Any]) -> str:
    if "generated_text" in resp.keys():
        return str(resp["generated_text"])
    else:
        raise ValueError('JSON response does not have "generated_keys" parameter')


class LLMOverREST(LLM):
    api_endpoint: str
    """The REST API endpoint from where the LLM will be served."""
    method: str = "POST"
    """The HTTP Method to use."""
    headers: Optional[Dict[str, str]] = None
    """A key-value map for all the request headers."""
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
    generated_text_key: str = "generated_text"
    """A key in the JSON response that corresponds to the generated text from the LLM. 
    This is used to extract the generated text and apply the stop sequence on it. 
    If this is ``None``, the ``stop`` parameter in the LLM call will not be honored. 
    Note: currently, it does not support nested keys and expect the keys to be 
    available at the root level of the response JSON"""

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        json_body = {"prompt": prompt}
        json_body.update(self.model_kwargs)

        if len(stop) > 0 and len(self.generated_text_key) == 0:
            raise ValueError(
                "``stop`` argument cannot be honored if "
                "``self.generated_text_key`` is None"
            )

        with requests.Session() as session:  # type: requests.Session
            try:
                response = session.request(
                    method=self.method,
                    url=self.api_endpoint,
                    headers=self.headers,
                    json=json_body,
                    proxies=self.proxies,
                    verify=self.ssl_verify,
                    cert=self.client_cert,
                )
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Error in invoking the REST API: {e}")

            if not response.ok:
                raise ValueError(
                    f"Error response from the API: "
                    f"{response.status_code}: {response.text}"
                )

            final_resp: str = ""
            if self.generated_text_key is not None:
                try:
                    resp_json = response.json()
                    gen_text = resp_json.get(self.generated_text_key)
                    if stop is not None and gen_text is not None:
                        gen_text = enforce_stop_tokens(gen_text, stop)
                        resp_json[self.generated_text_key] = gen_text
                        final_resp = json.dumps(resp_json)
                except requests.exceptions.JSONDecodeError as e:
                    raise ValueError(f"Error in parsing the response JSON: {e}")
            return final_resp if len(final_resp) > 0 else response.text

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
