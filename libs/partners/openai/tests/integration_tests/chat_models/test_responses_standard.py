"""Standard LangChain interface tests for Responses API"""

import base64
from pathlib import Path
from typing import cast

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_openai import ChatOpenAI
from tests.integration_tests.chat_models.test_base_standard import TestOpenAIStandard

REPO_ROOT_DIR = Path(__file__).parents[6]


class TestOpenAIResponses(TestOpenAIStandard):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "gpt-4o-mini", "use_responses_api": True}

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

    @property
    def supports_pdf_tool_message(self) -> bool:
        # OpenAI requires a filename for PDF inputs
        # For now, we test with filename in OpenAI-specific tests
        return False

    def test_openai_pdf_tool_messages(self, model: BaseChatModel) -> None:
        """Test that the model can process PDF inputs in ToolMessages."""
        url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        pdf_data = base64.b64encode(httpx.get(url).content).decode("utf-8")

        tool_message = ToolMessage(
            content=[
                {
                    "type": "file",
                    "source_type": "base64",
                    "data": pdf_data,
                    "mime_type": "application/pdf",
                    "filename": "my-pdf",  # specify filename
                },
            ],
            tool_call_id="1",
            name="random_pdf",
        )

        messages = [
            HumanMessage(
                "Get a random PDF using the tool and relay the title verbatim."
            ),
            AIMessage(
                [],
                tool_calls=[
                    {
                        "type": "tool_call",
                        "id": "1",
                        "name": "random_pdf",
                        "args": {},
                    }
                ],
            ),
            tool_message,
        ]

        def random_pdf() -> str:
            """Return a random PDF."""
            return ""

        _ = model.bind_tools([random_pdf]).invoke(messages)


def _invoke(llm: ChatOpenAI, input_: str, stream: bool) -> AIMessage:
    if stream:
        full = None
        for chunk in llm.stream(input_):
            full = full + chunk if full else chunk  # type: ignore[operator]
        return cast(AIMessage, full)
    return cast(AIMessage, llm.invoke(input_))
