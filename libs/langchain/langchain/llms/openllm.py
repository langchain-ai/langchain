from __future__ import annotations

import copy
import json
import logging
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    overload,
)

from langchain_core.pydantic_v1 import PrivateAttr

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import LLM

if TYPE_CHECKING:
    import openllm


ServerType = Literal["http"]


class IdentifyingParams(TypedDict):
    """Parameters for identifying a model as a typed dict."""

    model_name: str
    model_id: Optional[str]
    server_url: Optional[str]
    server_type: Optional[ServerType]
    embedded: bool
    llm_kwargs: Dict[str, Any]


logger = logging.getLogger(__name__)


class OpenLLM(LLM):
    """OpenLLM, supporting both in-process model
    instance and remote OpenLLM servers.

    To use, you should have the openllm library installed:

    .. code-block:: bash

        pip install openllm

    Learn more at: https://github.com/bentoml/openllm

    Example running an LLM model locally managed by OpenLLM:
        .. code-block:: python

            from langchain.llms import OpenLLM
            llm = OpenLLM(
                model_name='flan-t5',
                model_id='google/flan-t5-large',
            )
            llm("What is the difference between a duck and a goose?")

    For all available supported models, you can run 'openllm models'.

    If you have a OpenLLM server running, you can also use it remotely:
        .. code-block:: python

            from langchain.llms import OpenLLM
            llm = OpenLLM(server_url='http://localhost:3000')
            llm("What is the difference between a duck and a goose?")
    """

    model_name: Optional[str] = None
    """Model name to use. See 'openllm models' for all available models."""
    model_id: Optional[str] = None
    """Model Id to use. If not provided, will use the default model for the model name.
    See 'openllm models' for all available model variants."""
    server_url: Optional[str] = None
    """Optional server URL that currently runs a LLMServer with 'openllm start'."""
    server_type: ServerType = "http"
    """Optional server type. Either 'http' or 'grpc'."""
    embedded: bool = True
    """Initialize this LLM instance in current process by default. Should
    only set to False when using in conjunction with BentoML Service."""
    llm_kwargs: Dict[str, Any]
    """Keyword arguments to be passed to openllm.LLM"""

    _llm: Optional[openllm.LLM[Any, Any]] = PrivateAttr(default=None)
    _client: Optional[openllm.HTTPClient] = PrivateAttr(default=None)
    _async_client: Optional[openllm.AsyncHTTPClient] = PrivateAttr(default=None)

    class Config:
        extra = "forbid"

    @overload
    def __init__(
        self,
        model_name: Optional[str] = ...,
        *,
        model_id: Optional[str] = ...,
        embedded: Literal[True, False] = ...,
        **llm_kwargs: Any,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        *,
        server_url: str = ...,
        server_type: Literal["http"] = ...,
        **llm_kwargs: Any,
    ) -> None:
        ...

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        model_id: Optional[str] = None,
        server_url: Optional[str] = None,
        server_type: Literal["http"] = "http",
        embedded: bool = True,
        **llm_kwargs: Any,
    ):
        try:
            import openllm
        except ImportError as e:
            raise ImportError(
                "Could not import openllm. Make sure to install it with "
                "'pip install openllm.'"
            ) from e

        llm_kwargs = llm_kwargs or {}

        if server_url is not None:
            logger.debug("'server_url' is provided, returning a openllm.Client")
            assert (
                model_id is None and model_name is None
            ), "'server_url' and {'model_id', 'model_name'} are mutually exclusive"

            super().__init__(
                **{
                    "server_url": server_url,
                    "server_type": server_type,
                    "llm_kwargs": llm_kwargs,
                }
            )
            self._llm = None  # type: ignore
            self._client = openllm.HTTPClient(server_url)
            self._async_client = openllm.AsyncHTTPClient(server_url)
        else:
            if model_name is None:  # suports not passing model_name
                assert model_id is not None, "Must provide 'model_id' or 'server_url'"
                llm = openllm.LLM[t.Any, t.Any](model_id, embedded=embedded)
            else:
                assert (
                    model_name is not None
                ), "Must provide 'model_name' or 'server_url'"
                config = openllm.AutoConfig.for_model(model_name, **llm_kwargs)
                model_id = model_id or config["default_id"]
                # since the LLM are relatively huge, we don't actually want to convert the
                # Runner with embedded when running the server. Instead, we will only set
                # the init_local here so that LangChain users can still use the LLM
                # in-process. Wrt to BentoML users, setting embedded=False is the expected
                # behaviour to invoke the runners remotely.
                # We need to also enable ensure_available to download and setup the model.
                llm = openllm.LLM[Any, Any](
                    model_id, llm_config=config, embedded=embedded
                )
            super().__init__(
                **{
                    "model_name": model_name,
                    "model_id": model_id,
                    "embedded": embedded,
                    "llm_kwargs": llm_kwargs,
                }
            )
            self._client = None  # type: ignore
            self._async_client = None  # type: ignore
            self._llm = llm

    @property
    def runner(self) -> openllm.LLMRunner[Any, Any]:
        """
        Get the underlying openllm.LLMRunner instance for integration with BentoML.

        Example:
        .. code-block:: python

            llm = OpenLLM(
                model_name='flan-t5',
                model_id='google/flan-t5-large',
                embedded=False,
            )
            tools = load_tools(["serpapi", "llm-math"], llm=llm)
            agent = initialize_agent(
                tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
            )
            svc = bentoml.Service("langchain-openllm", runners=[llm.runner])

            @svc.api(input=Text(), output=Text())
            def chat(input_text: str):
                return agent.run(input_text)
        """
        warnings.warn(
            "'OpenLLM.runner' is deprecated, use 'OpenLLM.llm' instead",
            DeprecationWarning,
        )
        if self._llm is None:
            raise ValueError("OpenLLM must be initialized locally with 'model_name'")
        return self._llm.runner

    @property
    def llm(self) -> openllm.LLM[Any, Any]:
        """Get the underlying openllm.LLM instance."""
        if self._llm is None:
            raise ValueError("OpenLLM must be initialized locally with 'model_name'")
        return self._llm

    @property
    def _identifying_params(self) -> IdentifyingParams:
        """Get the identifying parameters."""
        if self._client is not None:
            self.llm_kwargs.update(self._client._config)
            model_name = self._client._metadata["model_name"]
            model_id = self._client._metadata["model_id"]
        else:
            if self._llm is None:
                raise ValueError("LLM must be initialized.")
            model_name = self.model_name
            model_id = self.model_id
            try:
                self.llm_kwargs.update(
                    json.loads(self._llm.identifying_params["configuration"])
                )
            except (TypeError, json.JSONDecodeError):
                pass
        return IdentifyingParams(
            server_url=self.server_url,
            server_type=self.server_type,
            embedded=self.embedded,
            llm_kwargs=self.llm_kwargs,
            model_name=model_name,
            model_id=model_id,
        )

    @property
    def _llm_type(self) -> str:
        return "openllm_client" if self._client else "openllm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        import asyncio

        try:
            import openllm
        except ImportError as e:
            raise ImportError(
                "Could not import openllm. Make sure to install it with "
                "'pip install openllm'."
            ) from e

        copied = copy.deepcopy(self.llm_kwargs)
        copied.update(kwargs)
        config = openllm.AutoConfig.for_model(
            self._identifying_params["model_name"], **copied
        )
        if self._client:
            res = self._client.generate(
                prompt, llm_config=config.model_dump(flatten=True), stop=stop
            )
        else:
            assert self._llm is not None
            res = asyncio.run(
                self._llm.generate(prompt, stop=stop, **config.model_dump(flatten=True))
            )
        if hasattr(res, "outputs"):
            return res.outputs[0].text
        else:
            raise ValueError(
                "Expected result to be either a 'openllm.GenerationOutput' or "
                f"'openllm_client.Response' output. Received '{res}' instead"
            )

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            import openllm
        except ImportError as e:
            raise ImportError(
                "Could not import openllm. Make sure to install it with "
                "'pip install openllm'."
            ) from e

        copied = copy.deepcopy(self.llm_kwargs)
        copied.update(kwargs)
        config = openllm.AutoConfig.for_model(
            self._identifying_params["model_name"], **copied
        )
        if self._async_client:
            res = await self._async_client.generate(
                prompt, llm_config=config.model_dump(flatten=True), stop=stop
            )
        else:
            assert self._llm is not None
            res = await self._llm.generate(
                prompt, stop=stop, **config.model_dump(flatten=True)
            )

        if hasattr(res, "outputs"):
            return res.outputs[0].text
        else:
            raise ValueError(
                "Expected result to be either a 'openllm.GenerationOutput' or "
                f"'openllm_client.Response' output. Received '{res}' instead"
            )
