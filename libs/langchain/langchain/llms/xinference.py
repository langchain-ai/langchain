from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Union

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

if TYPE_CHECKING:
    from xinference.client import RESTfulChatModelHandle, RESTfulGenerateModelHandle
    from xinference.model.llm.core import LlamaCppGenerateConfig
else:
    RESTfulGenerateModelHandle = Any
    RESTfulChatModelHandle = Any
    LlamaCppGenerateConfig = Any


class Xinference(LLM):
    """Wrapper for accessing Xinference's large-scale model inference service.
    To use, you should have the xinference library installed:
    .. code-block:: bash
        pip install "xinference[all]"
    Check out: https://github.com/xorbitsai/inference
    To run, you need to start a Xinference supervisor on one server and Xinference workers on the other servers
    Example:
        Starting the supervisor:
        .. code-block:: bash
            $ xinference-supervisor
        Starting the worker:
        .. code-block:: bash
            $ xinference-worker
    Then, you can accessing Xinference's model inference service.
    Example:
        .. code-block:: python

        from langchain.llms import Xinference
        llm = Xinference(
            server_url="http://0.0.0.0:9997",
            model_name="orca",
            model_size_in_billions=3,
            quantization="q4_0",
        )
        llm("Q: what is the capital of France? A:")
    To view all the supported builtin models, run:
    .. code-block:: bash
        $ xinference list --all
    """

    client: Any
    server_url: Optional[str]
    model_uid: Optional[str]
    """UID of the launched model"""

    def __init__(
        self, server_url: Optional[str] = None, model_uid: Optional[str] = None
    ):
        try:
            from xinference.client import RESTfulClient
        except ImportError as e:
            raise ImportError(
                "Could not import RESTfulClient from xinference. Make sure to install xinference in advance"
            ) from e

        super().__init__(
            **{
                "server_url": server_url,
                "model_uid": model_uid,
            }
        )

        if self.server_url is None:
            raise ValueError(f"Please provide server URL")

        if self.model_uid is None:
            raise ValueError(f"Please provide the model UID")

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
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        generate_config: Optional[LlamaCppGenerateConfig] = None,
    ) -> str:
        """ """
        model = self.client.get_model(self.model_uid)

        if stop:
            generate_config["stop"] = stop

        if generate_config and generate_config.get("stream") == True:
            combined_text_output = ""
            for token in self._stream(
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

    def _stream(
        self,
        model: Union[RESTfulGenerateModelHandle, RESTfulChatModelHandle],
        prompt: str,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        generate_config: Optional[LlamaCppGenerateConfig] = None,
    ):
        """ """
        streaming_response = model.generate(
            prompt=prompt, generate_config=generate_config
        )
        for chunk in streaming_response:
            if isinstance(chunk, dict):
                choices = chunk.get("choices", [])
                if choices:
                    choice = choices[0]
                    if isinstance(choice, dict):
                        token = choice.get("text")
                        log_probs = choice.get("logprobs")
                        if run_manager:
                            run_manager.on_llm_new_token(
                                token=token, verbose=self.verbose, log_probs=log_probs
                            )
                        yield token
