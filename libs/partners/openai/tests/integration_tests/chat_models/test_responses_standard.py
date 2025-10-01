"""Standard LangChain interface tests for Responses API"""

from pathlib import Path
from typing import cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain_openai import ChatOpenAI
from tests.integration_tests.chat_models.test_base_standard import TestOpenAIStandard

REPO_ROOT_DIR = Path(__file__).parents[6]


class TestOpenAIResponses(TestOpenAIStandard):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "gpt-4o-mini", "output_version": "responses/v1"}

    @property
    def supports_image_tool_message(self) -> bool:
        return True

    @pytest.mark.xfail(reason="Unsupported.")
    def test_stop_sequence(self, model: BaseChatModel) -> None:
        super().test_stop_sequence(model)

    def invoke_with_cache_read_input(self, *, stream: bool = False) -> AIMessage:
        with Path.open(REPO_ROOT_DIR / "README.md") as f:
            readme = f.read()

        input_ = f"""What's langchain? Here's the langchain README:

        {readme}
        """
        llm = ChatOpenAI(model="gpt-4.1-mini", output_version="responses/v1")
        _invoke(llm, input_, stream)
        # invoke twice so first invocation is cached
        return _invoke(llm, input_, stream)

    def invoke_with_reasoning_output(self, *, stream: bool = False) -> AIMessage:
        llm = ChatOpenAI(
            model="o4-mini",
            reasoning={"effort": "medium", "summary": "auto"},
            output_version="responses/v1",
        )
        input_ = "What was the 3rd highest building in 2000?"
        return _invoke(llm, input_, stream)

    def test_openai_pdf_inputs(self, model: BaseChatModel) -> None:
        """Test that the model can process PDF inputs."""
        super().test_openai_pdf_inputs(model)
        # Responses API additionally supports files via URL
        url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

        message = HumanMessage(
            [
                {"type": "text", "text": "What is the document title, verbatim?"},
                {"type": "file", "url": url},
            ]
        )
        _ = model.invoke([message])

        # Test OpenAI Responses format
        message = HumanMessage(
            [
                {"type": "text", "text": "What is the document title, verbatim?"},
                {"type": "input_file", "file_url": url},
            ]
        )
        _ = model.invoke([message])


def _invoke(llm: ChatOpenAI, input_: str, stream: bool) -> AIMessage:
    if stream:
        full = None
        for chunk in llm.stream(input_):
            full = full + chunk if full else chunk  # type: ignore[operator]
        return cast(AIMessage, full)
    return cast(AIMessage, llm.invoke(input_))
