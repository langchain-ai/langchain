import base64
import json
from typing import List, Optional

import httpx
import pytest
from langchain_core.language_models import BaseChatModel, GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import Field as FieldV1

from langchain_standard_tests.unit_tests.chat_models import (
    ChatModelTests,
    my_adder_tool,
)
from langchain_standard_tests.utils.pydantic import PYDANTIC_MAJOR_VERSION


class MagicFunctionSchema(BaseModel):
    input: int = Field(..., gt=-1000, lt=1000)


@tool(args_schema=MagicFunctionSchema)
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


@tool
def magic_function_no_args() -> int:
    """Calculates a magic function."""
    return 5


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


def _validate_tool_call_message(message: BaseMessage) -> None:
    assert isinstance(message, AIMessage)
    assert len(message.tool_calls) == 1
    tool_call = message.tool_calls[0]
    assert tool_call["name"] == "magic_function"
    assert tool_call["args"] == {"input": 3}
    assert tool_call["id"] is not None
    assert tool_call["type"] == "tool_call"


def _validate_tool_call_message_no_args(message: BaseMessage) -> None:
    assert isinstance(message, AIMessage)
    assert len(message.tool_calls) == 1
    tool_call = message.tool_calls[0]
    assert tool_call["name"] == "magic_function_no_args"
    assert tool_call["args"] == {}
    assert tool_call["id"] is not None
    assert tool_call["type"] == "tool_call"


class ChatModelIntegrationTests(ChatModelTests):
    @property
    def standard_chat_model_params(self) -> dict:
        return {}

    def test_invoke(self, model: BaseChatModel) -> None:
        result = model.invoke("Hello")
        assert result is not None
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, str)
        assert len(result.content) > 0

    async def test_ainvoke(self, model: BaseChatModel) -> None:
        result = await model.ainvoke("Hello")
        assert result is not None
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, str)
        assert len(result.content) > 0

    def test_stream(self, model: BaseChatModel) -> None:
        num_tokens = 0
        for token in model.stream("Hello"):
            assert token is not None
            assert isinstance(token, AIMessageChunk)
            num_tokens += len(token.content)
        assert num_tokens > 0

    async def test_astream(self, model: BaseChatModel) -> None:
        num_tokens = 0
        async for token in model.astream("Hello"):
            assert token is not None
            assert isinstance(token, AIMessageChunk)
            num_tokens += len(token.content)
        assert num_tokens > 0

    def test_batch(self, model: BaseChatModel) -> None:
        batch_results = model.batch(["Hello", "Hey"])
        assert batch_results is not None
        assert isinstance(batch_results, list)
        assert len(batch_results) == 2
        for result in batch_results:
            assert result is not None
            assert isinstance(result, AIMessage)
            assert isinstance(result.content, str)
            assert len(result.content) > 0

    async def test_abatch(self, model: BaseChatModel) -> None:
        batch_results = await model.abatch(["Hello", "Hey"])
        assert batch_results is not None
        assert isinstance(batch_results, list)
        assert len(batch_results) == 2
        for result in batch_results:
            assert result is not None
            assert isinstance(result, AIMessage)
            assert isinstance(result.content, str)
            assert len(result.content) > 0

    def test_conversation(self, model: BaseChatModel) -> None:
        messages = [
            HumanMessage("hello"),
            AIMessage("hello"),
            HumanMessage("how are you"),
        ]
        result = model.invoke(messages)
        assert result is not None
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, str)
        assert len(result.content) > 0

    def test_usage_metadata(self, model: BaseChatModel) -> None:
        if not self.returns_usage_metadata:
            pytest.skip("Not implemented.")
        result = model.invoke("Hello")
        assert result is not None
        assert isinstance(result, AIMessage)
        assert result.usage_metadata is not None
        assert isinstance(result.usage_metadata["input_tokens"], int)
        assert isinstance(result.usage_metadata["output_tokens"], int)
        assert isinstance(result.usage_metadata["total_tokens"], int)

    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        if not self.returns_usage_metadata:
            pytest.skip("Not implemented.")
        full: Optional[BaseMessageChunk] = None
        for chunk in model.stream("Hello"):
            assert isinstance(chunk, AIMessageChunk)
            full = chunk if full is None else full + chunk
        assert isinstance(full, AIMessageChunk)
        assert full.usage_metadata is not None
        assert isinstance(full.usage_metadata["input_tokens"], int)
        assert isinstance(full.usage_metadata["output_tokens"], int)
        assert isinstance(full.usage_metadata["total_tokens"], int)

    def test_stop_sequence(self, model: BaseChatModel) -> None:
        result = model.invoke("hi", stop=["you"])
        assert isinstance(result, AIMessage)

        custom_model = self.chat_model_class(
            **{**self.chat_model_params, "stop": ["you"]}
        )
        result = custom_model.invoke("hi")
        assert isinstance(result, AIMessage)

    def test_tool_calling(self, model: BaseChatModel) -> None:
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")
        if self.tool_choice_value == "tool_name":
            tool_choice: Optional[str] = "magic_function"
        else:
            tool_choice = self.tool_choice_value
        model_with_tools = model.bind_tools([magic_function], tool_choice=tool_choice)

        # Test invoke
        query = "What is the value of magic_function(3)? Use the tool."
        result = model_with_tools.invoke(query)
        _validate_tool_call_message(result)

        # Test stream
        full: Optional[BaseMessageChunk] = None
        for chunk in model_with_tools.stream(query):
            full = chunk if full is None else full + chunk  # type: ignore
        assert isinstance(full, AIMessage)
        _validate_tool_call_message(full)

    def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        if self.tool_choice_value == "tool_name":
            tool_choice: Optional[str] = "magic_function_no_args"
        else:
            tool_choice = self.tool_choice_value
        model_with_tools = model.bind_tools(
            [magic_function_no_args], tool_choice=tool_choice
        )
        query = "What is the value of magic_function()? Use the tool."
        result = model_with_tools.invoke(query)
        _validate_tool_call_message_no_args(result)

        full: Optional[BaseMessageChunk] = None
        for chunk in model_with_tools.stream(query):
            full = chunk if full is None else full + chunk  # type: ignore
        assert isinstance(full, AIMessage)
        _validate_tool_call_message_no_args(full)

    def test_bind_runnables_as_tools(self, model: BaseChatModel) -> None:
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        prompt = ChatPromptTemplate.from_messages(
            [("human", "Hello. Please respond in the style of {answer_style}.")]
        )
        llm = GenericFakeChatModel(messages=iter(["hello matey"]))
        chain = prompt | llm | StrOutputParser()
        tool_ = chain.as_tool(
            name="greeting_generator",
            description="Generate a greeting in a particular style of speaking.",
        )
        if self.tool_choice_value == "tool_name":
            tool_choice: Optional[str] = "greeting_generator"
        else:
            tool_choice = self.tool_choice_value
        model_with_tools = model.bind_tools([tool_], tool_choice=tool_choice)
        query = "Using the tool, generate a Pirate greeting."
        result = model_with_tools.invoke(query)
        assert isinstance(result, AIMessage)
        assert result.tool_calls
        tool_call = result.tool_calls[0]
        assert tool_call["args"].get("answer_style")
        assert tool_call["type"] == "tool_call"

    def test_structured_output(self, model: BaseChatModel) -> None:
        """Test to verify structured output with a Pydantic model."""
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        # Pydantic class
        # Type ignoring since the interface only officially supports pydantic 1
        # or pydantic.v1.BaseModel but not pydantic.BaseModel from pydantic 2.
        # We'll need to do a pass updating the type signatures.
        chat = model.with_structured_output(Joke)  # type: ignore[arg-type]
        result = chat.invoke("Tell me a joke about cats.")
        assert isinstance(result, Joke)

        for chunk in chat.stream("Tell me a joke about cats."):
            assert isinstance(chunk, Joke)

        # Schema
        chat = model.with_structured_output(Joke.model_json_schema())
        result = chat.invoke("Tell me a joke about cats.")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"setup", "punchline"}

        for chunk in chat.stream("Tell me a joke about cats."):
            assert isinstance(chunk, dict)
        assert isinstance(chunk, dict)  # for mypy
        assert set(chunk.keys()) == {"setup", "punchline"}

    async def test_structured_output_async(self, model: BaseChatModel) -> None:
        """Test to verify structured output with a Pydantic model."""
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        # Pydantic class
        # Type ignoring since the interface only officially supports pydantic 1
        # or pydantic.v1.BaseModel but not pydantic.BaseModel from pydantic 2.
        # We'll need to do a pass updating the type signatures.
        chat = model.with_structured_output(Joke)  # type: ignore[arg-type]
        result = await chat.ainvoke("Tell me a joke about cats.")
        assert isinstance(result, Joke)

        async for chunk in chat.astream("Tell me a joke about cats."):
            assert isinstance(chunk, Joke)

        # Schema
        chat = model.with_structured_output(Joke.model_json_schema())
        result = await chat.ainvoke("Tell me a joke about cats.")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"setup", "punchline"}

        async for chunk in chat.astream("Tell me a joke about cats."):
            assert isinstance(chunk, dict)
        assert isinstance(chunk, dict)  # for mypy
        assert set(chunk.keys()) == {"setup", "punchline"}

    @pytest.mark.skipif(PYDANTIC_MAJOR_VERSION != 2, reason="Test requires pydantic 2.")
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        """Test to verify compatibility with pydantic.v1.BaseModel.

        pydantic.v1.BaseModel is available in the pydantic 2 package.
        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        class Joke(BaseModelV1):  # Uses langchain_core.pydantic_v1.BaseModel
            """Joke to tell user."""

            setup: str = FieldV1(description="question to set up a joke")
            punchline: str = FieldV1(description="answer to resolve the joke")

        # Pydantic class
        chat = model.with_structured_output(Joke)
        result = chat.invoke("Tell me a joke about cats.")
        assert isinstance(result, Joke)

        for chunk in chat.stream("Tell me a joke about cats."):
            assert isinstance(chunk, Joke)

        # Schema
        chat = model.with_structured_output(Joke.schema())
        result = chat.invoke("Tell me a joke about cats.")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"setup", "punchline"}

        for chunk in chat.stream("Tell me a joke about cats."):
            assert isinstance(chunk, dict)
        assert isinstance(chunk, dict)  # for mypy
        assert set(chunk.keys()) == {"setup", "punchline"}

    def test_structured_output_optional_param(self, model: BaseChatModel) -> None:
        """Test to verify structured output with an optional param."""
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        class Joke(BaseModel):
            """Joke to tell user."""

            setup: str = Field(description="question to set up a joke")
            punchline: Optional[str] = Field(
                default=None, description="answer to resolve the joke"
            )

        chat = model.with_structured_output(Joke)  # type: ignore[arg-type]
        setup_result = chat.invoke(
            "Give me the setup to a joke about cats, no punchline."
        )
        assert isinstance(setup_result, Joke)

        joke_result = chat.invoke("Give me a joke about cats, include the punchline.")
        assert isinstance(joke_result, Joke)

    def test_tool_message_histories_string_content(self, model: BaseChatModel) -> None:
        """
        Test that message histories are compatible with string tool contents
        (e.g. OpenAI).
        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")
        model_with_tools = model.bind_tools([my_adder_tool])
        function_name = "my_adder_tool"
        function_args = {"a": "1", "b": "2"}

        messages_string_content = [
            HumanMessage("What is 1 + 2"),
            # string content (e.g. OpenAI)
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": function_name,
                        "args": function_args,
                        "id": "abc123",
                        "type": "tool_call",
                    },
                ],
            ),
            ToolMessage(
                json.dumps({"result": 3}),
                name=function_name,
                tool_call_id="abc123",
            ),
        ]
        result_string_content = model_with_tools.invoke(messages_string_content)
        assert isinstance(result_string_content, AIMessage)

    def test_tool_message_histories_list_content(
        self,
        model: BaseChatModel,
    ) -> None:
        """
        Test that message histories are compatible with list tool contents
        (e.g. Anthropic).
        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")
        model_with_tools = model.bind_tools([my_adder_tool])
        function_name = "my_adder_tool"
        function_args = {"a": 1, "b": 2}

        messages_list_content = [
            HumanMessage("What is 1 + 2"),
            # List content (e.g., Anthropic)
            AIMessage(
                [
                    {"type": "text", "text": "some text"},
                    {
                        "type": "tool_use",
                        "id": "abc123",
                        "name": function_name,
                        "input": function_args,
                    },
                ],
                tool_calls=[
                    {
                        "name": function_name,
                        "args": function_args,
                        "id": "abc123",
                        "type": "tool_call",
                    },
                ],
            ),
            ToolMessage(
                json.dumps({"result": 3}),
                name=function_name,
                tool_call_id="abc123",
            ),
        ]
        result_list_content = model_with_tools.invoke(messages_list_content)
        assert isinstance(result_list_content, AIMessage)

    def test_structured_few_shot_examples(self, model: BaseChatModel) -> None:
        """
        Test that model can process few-shot examples with tool calls.
        """
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")
        model_with_tools = model.bind_tools([my_adder_tool], tool_choice="any")
        function_name = "my_adder_tool"
        function_args = {"a": 1, "b": 2}
        function_result = json.dumps({"result": 3})

        messages_string_content = [
            HumanMessage("What is 1 + 2"),
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": function_name,
                        "args": function_args,
                        "id": "abc123",
                        "type": "tool_call",
                    },
                ],
            ),
            ToolMessage(
                function_result,
                name=function_name,
                tool_call_id="abc123",
            ),
            AIMessage(function_result),
            HumanMessage("What is 3 + 4"),
        ]
        result_string_content = model_with_tools.invoke(messages_string_content)
        assert isinstance(result_string_content, AIMessage)

    def test_image_inputs(self, model: BaseChatModel) -> None:
        if not self.supports_image_inputs:
            return
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": "describe the weather in this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        )
        model.invoke([message])

    def test_image_tool_message(self, model: BaseChatModel) -> None:
        if not self.supports_image_tool_message:
            return
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
        messages = [
            HumanMessage("get a random image using the tool and describe the weather"),
            AIMessage(
                [],
                tool_calls=[
                    {"type": "tool_call", "id": "1", "name": "random_image", "args": {}}
                ],
            ),
            ToolMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
                tool_call_id="1",
                name="random_image",
            ),
        ]

        def random_image() -> str:
            """Return a random image."""
            return ""

        model.bind_tools([random_image]).invoke(messages)

    def test_anthropic_inputs(self, model: BaseChatModel) -> None:
        if not self.supports_anthropic_inputs:
            return

        class color_picker(BaseModelV1):
            """Input your fav color and get a random fact about it."""

            fav_color: str

        human_content: List[dict] = [
            {
                "type": "text",
                "text": "what's your favorite color in this image",
            },
        ]
        if self.supports_image_inputs:
            image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
            human_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                }
            )
        messages = [
            SystemMessage("you're a good assistant"),
            HumanMessage(human_content),  # type: ignore[arg-type]
            AIMessage(
                [
                    {"type": "text", "text": "Hmm let me think about that"},
                    {
                        "type": "tool_use",
                        "input": {"fav_color": "green"},
                        "id": "foo",
                        "name": "color_picker",
                    },
                ]
            ),
            HumanMessage(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "foo",
                        "content": [
                            {
                                "type": "text",
                                "text": "green is a great pick! that's my sister's favorite color",  # noqa: E501
                            }
                        ],
                        "is_error": False,
                    },
                    {"type": "text", "text": "what's my sister's favorite color"},
                ]
            ),
        ]
        model.bind_tools([color_picker]).invoke(messages)

    def test_tool_message_error_status(self, model: BaseChatModel) -> None:
        """Test that ToolMessage with status='error' can be handled."""
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")
        model_with_tools = model.bind_tools([my_adder_tool])
        messages = [
            HumanMessage("What is 1 + 2"),
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "my_adder_tool",
                        "args": {"a": 1},
                        "id": "abc123",
                        "type": "tool_call",
                    },
                ],
            ),
            ToolMessage(
                "Error: Missing required argument 'b'.",
                name="my_adder_tool",
                tool_call_id="abc123",
                status="error",
            ),
        ]
        result = model_with_tools.invoke(messages)
        assert isinstance(result, AIMessage)

    def test_message_with_name(self, model: BaseChatModel) -> None:
        result = model.invoke([HumanMessage("hello", name="example_user")])
        assert result is not None
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, str)
        assert len(result.content) > 0
