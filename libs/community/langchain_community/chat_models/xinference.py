import json
from typing import Any, Dict, Iterator, List, Optional, Union, cast, TYPE_CHECKING

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

if TYPE_CHECKING:
    from xinference.client import RESTfulChatModelHandle
    from xinference.model.llm.core import LlamaCppGenerateConfig


class ChatXinference(BaseChatModel):
    """`Xinference` large-scale model inference service.

    To use, you should have the xinference library installed:

    .. code-block:: bash

       pip install "xinference[all]"

    If you're simply using the services provided by Xinference, you can utilize the xinference_client package:

    .. code-block:: bash

        pip install xinference_client

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

    It will return a model UID. Then, you can use ChatXinference with LangChain.

    Example:

    .. code-block:: python

        from langchain_community.chat_models.xinference import ChatXinference

        llm = ChatXinference(
            server_url="http://0.0.0.0:9997",
            model_uid = {model_uid} # replace model_uid with the model UID return from launching the model
        )

        llm.invoke(
            input="Q: where can we visit in the capital of France? A:",
            generate_config={"max_tokens": 1024, "stream": True},
        )

    Example:

    .. code-block:: python

        from langchain_community.chat_models.xinference import ChatXinference
        from langchain.prompts import PromptTemplate

        llm = ChatXinference(
            server_url="http://0.0.0.0:9997",
            model_uid={model_uid}, # replace model_uid with the model UID return from launching the model
        )
        prompt = PromptTemplate(
            input=['country'],
            template="Q: where can we visit in the capital of {country}? A:"
        )
        chain = prompt | llm
        chain.invoke(input={'country': 'France'})

        chain.stream(input={'country': 'France'})  #  streaming data


    To view all the supported builtin models, run:

    .. code-block:: bash

        $ xinference list --all

    """  # noqa: E501

    client: Optional[Any] = None
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
        api_key: Optional[str] = None,
        **model_kwargs: Any,
    ):
        try:
            from xinference.client import RESTfulClient
        except ImportError:
            try:
                from xinference_client import RESTfulClient
            except ImportError as e:
                raise ImportError(
                    "Could not import RESTfulClient from xinference. Please install it"
                    " with `pip install xinference` or `pip install xinference_client`."
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

        self._headers: Dict[str, str] = {}
        self._cluster_authed = False
        self._check_cluster_authenticated()
        if api_key is not None and self._cluster_authed:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self.client = RESTfulClient(server_url)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "xinference-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"server_url": self.server_url},
            **{"model_uid": self.model_uid},
            **{"model_kwargs": self.model_kwargs},
        }

    def _check_cluster_authenticated(self) -> None:
        url = f"{self.server_url}/v1/cluster/auth"
        response = requests.get(url)
        if response.status_code == 404:
            self._cluster_authed = False
        else:
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to get cluster information, "
                    f"detail: {response.json()['detail']}"
                )
            response_data = response.json()

            self._cluster_authed = bool(response_data["auth"])

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.client is None:
            raise ValueError("Client is not initialized!")

        model = self.client.get_model(self.model_uid)
        generate_config: "LlamaCppGenerateConfig" = kwargs.get("generate_config", {})
        generate_config = {**self.model_kwargs, **generate_config}

        if stop:
            generate_config["stop"] = stop

        final_chunk = self._chat_with_aggregation(
            model=model,
            messages=messages,
            run_manager=run_manager,
            verbose=self.verbose,
            generate_config=generate_config,
        )
        chat_generation = ChatGeneration(
            message=AIMessage(content=final_chunk.text),
            generation_info=final_chunk.generation_info,
        )

        return ChatResult(generations=[chat_generation])

    def _chat_with_aggregation(
        self,
        model: Union["RESTfulChatModelHandle"],
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        generate_config: Optional["LlamaCppGenerateConfig"] = None,
    ) -> ChatGenerationChunk:
        response = model.chat(
            messages=self._create_message_dicts(messages),
            generate_config=generate_config,
        )

        final_chunk: Optional[ChatGenerationChunk] = None
        for stream_resp in response:
            if stream_resp:
                chunk = self._chat_response_to_chat_generation_chunk(
                    stream_resp["choices"][0]
                )
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from xinference stream.")

        return final_chunk

    @staticmethod
    def _chat_response_to_chat_generation_chunk(
        stream_response: Dict[str, Any],
    ) -> ChatGenerationChunk:
        generation_info = (
            stream_response if stream_response.get("finish_reason") == "stop" else None
        )
        return ChatGenerationChunk(
            message=AIMessageChunk(
                content=stream_response.get("delta", {}).get("content", "")
            ),
            generation_info=generation_info,
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if self.client is None:
            raise ValueError("Client is not initialized!")

        model = self.client.get_model(self.model_uid)

        generate_config = kwargs.get("generate_config", {})
        if "stream" not in generate_config or not generate_config.get("stream", False):
            generate_config["stream"] = True
        generate_config = {**self.model_kwargs, **generate_config}
        if stop:
            generate_config["stop"] = stop
        response = model.chat(
            messages=self._create_message_dicts(messages),
            generate_config=generate_config,
        )

        for stream_resp in response:
            if stream_resp:
                chunk = self._chat_response_to_chat_generation_chunk(
                    stream_resp["choices"][0]
                )
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk

    @staticmethod
    def _create_message_dicts(
        messages: List[BaseMessage],
    ) -> List[Dict[str, Union[str, List[str]]]]:
        messages_list: List = []
        for message in messages:
            role = ""
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                raise ValueError("Received unsupported message type.")

            content = ""
            if isinstance(message.content, str):
                content = message.content
            else:
                for content_part in cast(List[Dict], message.content):
                    if content_part.get("type") == "text":
                        content += f"\n{content_part['text']}"
                    else:
                        raise ValueError("Unsupported message content type. ")

            messages_list.append(
                {
                    "role": role,
                    "content": content,
                }
            )
        return messages_list
