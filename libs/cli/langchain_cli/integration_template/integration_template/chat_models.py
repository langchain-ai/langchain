"""__ModuleName__ chat models."""

from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


class Chat__ModuleName__(BaseChatModel):
    # TODO: Replace all TODOs in docstring. See example docstring:
    # https://github.com/langchain-ai/langchain/blob/7ff05357bac6eaedf5058a2af88f23a1817d40fe/libs/partners/openai/langchain_openai/chat_models/base.py#L1120
    """__ModuleName__ chat model integration.

    The default implementation echoes the first `parrot_buffer_length` characters of the input.

    # TODO: Replace with relevant packages, env vars.
    Setup:
        Install ``__package_name__`` and set environment variable ``__MODULE_NAME___API_KEY``.

        .. code-block:: bash

            pip install -U __package_name__
            export __MODULE_NAME___API_KEY="your-api-key"

    # TODO: Populate with relevant params.
    Key init args — completion params:
        model: str
            Name of __ModuleName__ model to use.
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

    # TODO: Populate with relevant params.
    Key init args — client params:
        timeout: Optional[float]
            Timeout for requests.
        max_retries: int
            Max number of retries.
        api_key: Optional[str]
            __ModuleName__ API key. If not passed in will be read from env var __MODULE_NAME___API_KEY.

    See full list of supported init args and their descriptions in the params section.

    # TODO: Replace with relevant init params.
    Instantiate:
        .. code-block:: python

            from __module_name__ import Chat__ModuleName__

            llm = Chat__ModuleName__(
                model="...",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                # api_key="...",
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if token-level streaming isn't supported.
    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk.text(), end="")

        .. code-block:: python

            # TODO: Example output.

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if native async isn't supported.
    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

            # stream:
            # async for chunk in (await llm.astream(messages))

            # batch:
            # await llm.abatch([messages])

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if .bind_tools() isn't supported.
    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field

            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

            class GetPopulation(BaseModel):
                '''Get the current population in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
            ai_msg = llm_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
            ai_msg.tool_calls

        .. code-block:: python

              # TODO: Example output.

        See ``Chat__ModuleName__.bind_tools()`` method for more.

    # TODO: Delete if .with_structured_output() isn't supported.
    Structured output:
        .. code-block:: python

            from typing import Optional

            from pydantic import BaseModel, Field

            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")
                rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")

            structured_llm = llm.with_structured_output(Joke)
            structured_llm.invoke("Tell me a joke about cats")

        .. code-block:: python

            # TODO: Example output.

        See ``Chat__ModuleName__.with_structured_output()`` for more.

    # TODO: Delete if JSON mode response format isn't supported.
    JSON mode:
        .. code-block:: python

            # TODO: Replace with appropriate bind arg.
            json_llm = llm.bind(response_format={"type": "json_object"})
            ai_msg = json_llm.invoke("Return a JSON object with key 'random_ints' and a value of 10 random ints in [0-99]")
            ai_msg.content

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if image inputs aren't supported.
    Image input:
        .. code-block:: python

            import base64
            import httpx
            from langchain_core.messages import HumanMessage

            image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
            # TODO: Replace with appropriate message content format.
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "describe the weather in this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            )
            ai_msg = llm.invoke([message])
            ai_msg.content

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if audio inputs aren't supported.
    Audio input:
        .. code-block:: python

            # TODO: Example input

        .. code-block:: python

            # TODO: Example output

    # TODO: Delete if video inputs aren't supported.
    Video input:
        .. code-block:: python

            # TODO: Example input

        .. code-block:: python

            # TODO: Example output

    # TODO: Delete if token usage metadata isn't supported.
    Token usage:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.usage_metadata

        .. code-block:: python

            {'input_tokens': 28, 'output_tokens': 5, 'total_tokens': 33}

    # TODO: Delete if logprobs aren't supported.
    Logprobs:
        .. code-block:: python

            # TODO: Replace with appropriate bind arg.
            logprobs_llm = llm.bind(logprobs=True)
            ai_msg = logprobs_llm.invoke(messages)
            ai_msg.response_metadata["logprobs"]

        .. code-block:: python

              # TODO: Example output.

    Response metadata
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

             # TODO: Example output.

    """  # noqa: E501

    model_name: str = Field(alias="model")
    """The name of the model"""
    parrot_buffer_length: int
    """The number of characters from the last message of the prompt to be echoed."""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-__package_name_short__"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        # Replace this with actual logic to generate a response from a list
        # of messages.
        last_message = messages[-1]
        tokens = last_message.content[: self.parrot_buffer_length]
        ct_input_tokens = sum(len(message.content) for message in messages)
        ct_output_tokens = len(tokens)
        message = AIMessage(
            content=tokens,
            additional_kwargs={},  # Used to add additional payload to the message
            response_metadata={  # Use for response metadata
                "time_in_seconds": 3,
            },
            usage_metadata={
                "input_tokens": ct_input_tokens,
                "output_tokens": ct_output_tokens,
                "total_tokens": ct_input_tokens + ct_output_tokens,
            },
        )
        ##

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model.

        This method should be implemented if the model can generate output
        in a streaming fashion. If the model does not support streaming,
        do not implement it. In that case streaming requests will be automatically
        handled by the _generate method.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        last_message = messages[-1]
        tokens = str(last_message.content[: self.parrot_buffer_length])
        ct_input_tokens = sum(len(message.content) for message in messages)

        for token in tokens:
            usage_metadata = UsageMetadata(
                {
                    "input_tokens": ct_input_tokens,
                    "output_tokens": 1,
                    "total_tokens": ct_input_tokens + 1,
                }
            )
            ct_input_tokens = 0
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=token, usage_metadata=usage_metadata)
            )

            if run_manager:
                # This is optional in newer versions of LangChain
                # The on_llm_new_token will be called automatically
                run_manager.on_llm_new_token(token, chunk=chunk)

            yield chunk

        # Let's add some other information (e.g., response metadata)
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={"time_in_sec": 3})
        )
        if run_manager:
            # This is optional in newer versions of LangChain
            # The on_llm_new_token will be called automatically
            run_manager.on_llm_new_token(token, chunk=chunk)
        yield chunk

    # TODO: Implement if Chat__ModuleName__ supports async streaming. Otherwise delete.
    # async def _astream(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> AsyncIterator[ChatGenerationChunk]:

    # TODO: Implement if Chat__ModuleName__ supports async generation. Otherwise delete.
    # async def _agenerate(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> ChatResult:
