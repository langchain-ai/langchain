from typing import Any, Dict, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.utils import pre_init
from pydantic import ConfigDict


class ManifestWrapper(LLM):
    """HazyResearch's Manifest library."""

    client: Any = None  #: :meta private:
    llm_kwargs: Optional[Dict] = None

    model_config = ConfigDict(
        extra="forbid",
    )

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""
        try:
            from manifest import Manifest

            if not isinstance(values["client"], Manifest):
                raise ValueError
        except ImportError:
            raise ImportError(
                "Could not import manifest python package. "
                "Please install it with `pip install manifest-ml`."
            )
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        kwargs = self.llm_kwargs or {}
        return {
            **self.client.client_pool.get_current_client().get_model_params(),
            **kwargs,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "manifest"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to LLM through Manifest."""
        if stop is not None and len(stop) != 1:
            raise NotImplementedError(
                f"Manifest currently only supports a single stop token, got {stop}"
            )
        params = self.llm_kwargs or {}
        params = {**params, **kwargs}
        if stop is not None:
            params["stop_token"] = stop
        return self.client.run(prompt, **params)
