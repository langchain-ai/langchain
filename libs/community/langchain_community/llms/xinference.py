from typing import TYPE_CHECKING, Any, Dict, Generator, List, Mapping, Optional, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

if TYPE_CHECKING:
    from xinference.client import RESTfulChatModelHandle, RESTfulGenerateModelHandle
    from xinference.model.llm.core import LlamaCppGenerateConfig


class Xinference(LLM):
    """`Xinference` large-scale model inference service.

    To use, you should have the xinference library installed:

    .. code-block:: bash

       pip install "xinference[all]"

    Check out: https://github.com/xorbitsai/inference
    To run, you need to start a Xinference supervisor on one server and Xinference workers on the other servers

    Example:
        To start a local instance of Xinference, run

        .. code-block:: bash

           $ xinference

        You can also deploy Xinference in a distributed cluster. Here are the steps:

        Starting the supervisor:

        .. code-block:: bash

           $ xinference-supervisor

        Starting the worker:

        .. code-block:: bash

           $ xinference-worker

    Then, launch a model using command line interface (CLI).

    Example:

    .. code-block:: bash

       $ xinference launch -n orca -s 3 -q q4_0

    It will return a model UID. Then, you can use Xinference with LangChain.

    Example:

    .. code-block:: python

        from langchain_community.llms import Xinference

        llm = Xinference(
            server_url="http://0.0.0.0:9997",
            model_uid = {model_uid} # replace model_uid with the model UID return from launching the model
        )

        llm.invoke(
            prompt="Q: where can we visit in the capital of France? A:",
            generate_config={"max_tokens": 1024, "stream": True},
        )

    To view all the supported builtin models, run:

    .. code-block:: bash

        $ xinference list --all

    """  # noqa: E501

    client: Any
    server_url: Optional[str]
    """URL of the xinference server"""
    model_uid: Optional[str]
    """UID of the launched model"""
    model_kwargs: Dict[str, Any]
    """Keyword arguments to be passed to xinference.LLM"""

    def __init__(
        self,
        server_url: Optional[str] = None,
        model_uid: Optional[str] = None,
        **model_kwargs: Any,
    ):
        try:
            from xinference.client import RESTfulClient
        except ImportError as e:
            raise ImportError(
                "Could not import RESTfulClient from xinference. Please install it"
                " with `pip install xinference`."
            ) from e

        model_kwargs = model_kwargs or {}

        super().__init__(
            **{  # type: ignore[arg-type]
                "server_url": server_url,
                "model_uid": model_uid,
                "model_kwargs": model_kwargs,
            }
        )

        if self.server_url is None:
            raise ValueError("Please provide server URL")

        if self.model_uid is None:
            raise ValueError("Please provide the model UID")

        self.client = RESTfulClient(server_url)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "xinference"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"server_url": self.server_url},
            **{"model_uid": self.model_uid},
            **{"model_kwargs": self.model_kwargs},
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the xinference model and return the output.

        Args:
            prompt: The prompt to use for generation.
            stop: Optional list of stop words to use when generating.
            generate_config: Optional dictionary for the configuration used for
                generation.

        Returns:
            The generated string by the model.
        """
        model = self.client.get_model(self.model_uid)

        generate_config: "LlamaCppGenerateConfig" = kwargs.get("generate_config", {})

        generate_config = {**self.model_kwargs, **generate_config}

        if stop:
            generate_config["stop"] = stop

        if generate_config and generate_config.get("stream"):
            combined_text_output = ""
            for token in self._stream_generate(
                model=model,
                prompt=prompt,
                run_manager=run_manager,
                generate_config=generate_config,
            ):
                combined_text_output += token
            return combined_text_output

        else:
            completion = model.generate(prompt=prompt, generate_config=generate_config)
            return completion["choices"][0]["text"]

    def _stream_generate(
        self,
        model: Union["RESTfulGenerateModelHandle", "RESTfulChatModelHandle"],
        prompt: str,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        generate_config: Optional["LlamaCppGenerateConfig"] = None,
    ) -> Generator[str, None, None]:
        """
        Args:
            prompt: The prompt to use for generation.
            model: The model used for generation.
            stop: Optional list of stop words to use when generating.
            generate_config: Optional dictionary for the configuration used for
                generation.

        Yields:
            A string token.
        """
        streaming_response = model.generate(
            prompt=prompt, generate_config=generate_config
        )
        for chunk in streaming_response:
            if isinstance(chunk, dict):
                choices = chunk.get("choices", [])
                if choices:
                    choice = choices[0]
                    if isinstance(choice, dict):
                        token = choice.get("text", "")
                        log_probs = choice.get("logprobs")
                        if run_manager:
                            run_manager.on_llm_new_token(
                                token=token, verbose=self.verbose, log_probs=log_probs
                            )
                        yield token
