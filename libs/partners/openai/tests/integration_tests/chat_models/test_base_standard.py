"""Standard LangChain interface tests"""

import base64
from pathlib import Path
from typing import Literal, cast

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_openai import ChatOpenAI

REPO_ROOT_DIR = Path(__file__).parents[6]


class TestOpenAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "gpt-4o-mini", "stream_usage": True}

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_urls(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def supports_anthropic_inputs(self) -> bool:
        return True

    @property
    def supported_usage_metadata_details(
        self,
    ) -> dict[
        Literal["invoke", "stream"],
        list[
            Literal[
                "audio_input",
                "audio_output",
                "reasoning_output",
                "cache_read_input",
                "cache_creation_input",
            ]
        ],
    ]:
        return {"invoke": ["reasoning_output", "cache_read_input"], "stream": []}

    @property
    def enable_vcr_tests(self) -> bool:
        return True

    def invoke_with_cache_read_input(self, *, stream: bool = False) -> AIMessage:
        with open(REPO_ROOT_DIR / "README.md") as f:
            readme = f.read()

        input_ = f"""What's langchain? Here's the langchain README:

        {readme}
        """
        llm = ChatOpenAI(model="gpt-4o-mini", stream_usage=True)
        _invoke(llm, input_, stream)
        # invoke twice so first invocation is cached
        return _invoke(llm, input_, stream)

    def invoke_with_reasoning_output(self, *, stream: bool = False) -> AIMessage:
        llm = ChatOpenAI(model="o1-mini", stream_usage=True, temperature=1)
        input_ = (
            "explain  the relationship between the 2008/9 economic crisis and the "
            "startup ecosystem in the early 2010s"
        )
        return _invoke(llm, input_, stream)

    @property
    def supports_pdf_inputs(self) -> bool:
        # OpenAI requires a filename for PDF inputs
        # For now, we test with filename in OpenAI-specific tests
        return False

    def test_openai_pdf_inputs(self, model: BaseChatModel) -> None:
        """Test that the model can process PDF inputs."""
        url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        pdf_data = base64.b64encode(httpx.get(url).content).decode("utf-8")

        message = HumanMessage(
            [
                {"type": "text", "text": "Summarize this document:"},
                {
                    "type": "file",
                    "source_type": "base64",
                    "mime_type": "application/pdf",
                    "data": pdf_data,
                    "filename": "my-pdf",  # OpenAI requires a filename
                },
            ]
        )
        _ = model.invoke([message])

        # Test OpenAI Chat Completions format
        message = HumanMessage(
            [
                {"type": "text", "text": "Summarize this document:"},
                {
                    "type": "file",
                    "file": {
                        "filename": "test file.pdf",
                        "file_data": f"data:application/pdf;base64,{pdf_data}",
                    },
                },
            ]
        )
        _ = model.invoke([message])


def _invoke(llm: ChatOpenAI, input_: str, stream: bool) -> AIMessage:
    if stream:
        full = None
        for chunk in llm.stream(input_):
            full = full + chunk if full else chunk  # type: ignore[operator]
        return cast(AIMessage, full)
    else:
        return cast(AIMessage, llm.invoke(input_))


@pytest.mark.skip()  # Test either finishes in 5 seconds or 5 minutes.
def test_audio_model() -> None:
    class AudioModelTests(ChatModelIntegrationTests):
        @property
        def chat_model_class(self) -> type[ChatOpenAI]:
            return ChatOpenAI

        @property
        def chat_model_params(self) -> dict:
            return {
                "model": "gpt-4o-audio-preview",
                "temperature": 0,
                "model_kwargs": {
                    "modalities": ["text", "audio"],
                    "audio": {"voice": "alloy", "format": "wav"},
                },
            }

        @property
        def supports_audio_inputs(self) -> bool:
            return True

    test_instance = AudioModelTests()
    model = test_instance.chat_model_class(**test_instance.chat_model_params)
    AudioModelTests().test_audio_inputs(model)
