from typing import Any, Dict, List, Optional

import requests
from langchain.chains.router.llm_router import RouterChain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import SecretStr, model_validator

JINA_API_URL: str = "http://api.jina.ai/v1/classify"


class JinaClassifierRouterChain(RouterChain):
    session: Any
    model_name: str
    routing_keys: List[str] = ["query"]
    jina_api_key: Optional[SecretStr] = None
    labels: Optional[List[str]] = None

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the LLM chain prompt expects.

        :meta private:
        """
        return self.routing_keys

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, values: Dict) -> Any:
        try:
            jina_api_key = convert_to_secret_str(
                get_from_dict_or_env(values, "jina_api_key", "JINA_API_KEY")
            )
        except ValueError as original_exc:
            try:
                jina_api_key = convert_to_secret_str(
                    get_from_dict_or_env(values, "jina_auth_token", "JINA_AUTH_TOKEN")
                )
            except ValueError:
                raise original_exc
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {jina_api_key.get_secret_value()}",
                "Accept-Encoding": "identity",
                "Content-type": "application/json",
            }
        )
        values["session"] = session
        return values

    def _classify(self, inputs: List[Dict[str, str]], labels: List[str]) -> List[str]:
        resp = self.session.post(  # type: ignore
            JINA_API_URL,
            json={"input": inputs, "labels": labels, "model": self.model_name},
        ).json()
        if "data" not in resp:
            raise RuntimeError(resp["detail"])

        results = resp["data"]

        # Sort resulting embeddings by index
        sorted_results = sorted(results, key=lambda e: e["index"])  # type: ignore

        # Return just the embeddings
        return [r["prediction"] for r in sorted_results]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _input = ", ".join(inputs[k] for k in self.routing_keys)
        _labels = inputs.get("labels", self.labels)
        results = self._classify(
            [
                {
                    "text": _input,
                },
            ],
            _labels,
        )
        return {"next_inputs": inputs, "destination": results[0]}
