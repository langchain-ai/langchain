"""Wrapper around OpenLLM APIs."""
from __future__ import annotations

import json
import logging
import typing as t

from langchain.llms.base import LLM

if t.TYPE_CHECKING:
    ServerType = t.Literal["http", "grpc"]
    from langchain.callbacks.manager import CallbackManagerForLLMRun
else:
    ServerType = str

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "flan-t5"


class OpenLLM(LLM):
    """Wrapper around the OpenLLM model.

    To use, you should have the openllm library installed, and
    provide the model name that you want to run as named parameter.

    For all available supported models, you can run 'openllm models'.

    Checkout: https://github.com/llmsys/openllm

    Example:
        .. code-block:: python

            from langchain.llms import OpenLLM
            llm = OpenLLM.for_model(model_name='flan-t5', **llm_attrs)
            llm("What is the difference between a duck and a goose?")

    If you have a OpenLLM server running somewhere, you can also do
    the following:

    Example:
        .. code-block:: python

            from langchain.llms import OpenLLM
            llm = OpenLLM.for_model(server_url='http://localhost:8000', server_type='http')
            llm("What is the difference between a duck and a goose?")
    """

    llm: t.Any  #: :meta private:
    model_name: str = DEFAULT_MODEL_NAME
    """Model name that use OpenLLM to run. Check 'openllm models' for list of supported models."""
    llm_attrs: t.Optional[t.Dict[str, t.Any]] = None
    """Key word arguments to be passed to openllm.LLM"""
    server_url: t.Optional[str] = None
    """Optional server URL that currently runs a LLMServer with 'openllm start'."""
    server_type: ServerType = "http"
    """Optional server type. Either 'http' or 'grpc'."""

    class Config:
        extra = "forbid"

    @classmethod
    def for_model(
        cls,
        model_name: str = DEFAULT_MODEL_NAME,
        model_id: str | None = None,
        **llm_attrs: t.Any,
    ) -> LLM:
        """Construct the pipeline object from model_id and task."""
        try:
            import openllm
        except ImportError:
            raise ValueError(
                "Could not import openllm. Make sure to install it with 'pip install openllm.'"
            )
        server_url = llm_attrs.pop("server_url", None)
        server_type: t.Literal["http", "grpc"] = llm_attrs.pop("server_type", "http")
        if server_url is not None:
            logger.debug("'server_url' is provided, returning a Client.")
            server_cls = (
                openllm.client.HTTPClient
                if server_type == "http"
                else openllm.client.GrpcClient
            )
            llm = server_cls(server_url)

            return cls(
                llm=llm,
                model_name=llm.model_name,
                server_type=server_type,
                server_url=server_url,
                llm_attrs=llm_attrs,
            )

        return cls(
            llm=openllm.Runner(
                model_name,
                model_id=model_id,
                init_local=True,
                **llm_attrs,
            ),
            model_name=model_name,
            llm_attrs=llm_attrs,
        )

    @property
    def _identifying_params(self) -> t.Mapping[str, t.Any]:
        """Get the identifying parameters."""
        res = {
            "model_name": self.model_name,
            "llm_attrs": self.llm_attrs,
            "server_url": self.server_url,
            "server_type": self.server_type,
        }
        if self._llm_type == "openllm_client":
            return res
        assert self.llm_attrs is not None
        try:
            self.llm_attrs.update(
                json.loads(self.llm.identifying_params["configuration"])
            )
        except (TypeError, json.JSONDecodeError):
            pass
        res["llm_attrs"] = self.llm_attrs
        return res

    @property
    def _llm_type(self) -> str:
        return "openllm_client" if self.server_url is not None else "openllm"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
    ) -> str:
        import openllm

        if self.server_url is not None:
            config = openllm.AutoConfig.for_model(
                self.llm.model_name, **(self.llm_attrs or {})
            )
        else:
            config = self.llm.config.model_construct_env(**(self.llm_attrs or {}))

        if self.server_url is not None:
            return self.llm.query(prompt, **config.model_dump(flatten=True))
        else:
            return self.llm(prompt, **config.model_dump(flatten=True))
