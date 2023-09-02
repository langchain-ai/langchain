from typing import Any, List, Optional

from pytest_mock import MockerFixture
from syrupy import SnapshotAssertion

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.runnables.openai_functions import OpenAIFunctionsRouter
from langchain.schema import ChatResult
from langchain.schema.messages import AIMessage, BaseMessage
from langchain.schema.output import ChatGeneration


class FakeChatOpenAI(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake-openai-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        additional_kwargs={
                            "function_call": {
                                "name": "accept",
                                "arguments": '{\n  "draft": "turtles"\n}',
                            }
                        },
                    )
                )
            ]
        )


def test_openai_functions_router(
    snapshot: SnapshotAssertion, mocker: MockerFixture
) -> None:
    revise = mocker.Mock(
        side_effect=lambda kw: f'Revised draft: no more {kw["notes"]}!'
    )
    accept = mocker.Mock(side_effect=lambda kw: f'Accepted draft: {kw["draft"]}!')

    router = OpenAIFunctionsRouter(
        {
            "revise": revise,
            "accept": accept,
        },
        functions=[
            {
                "name": "revise",
                "description": "Sends the draft for revision.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "notes": {
                            "type": "string",
                            "description": "The editor's notes to guide the revision.",
                        },
                    },
                },
            },
            {
                "name": "accept",
                "description": "Accepts the draft.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "draft": {
                            "type": "string",
                            "description": "The draft to accept.",
                        },
                    },
                },
            },
        ],
    )

    model = FakeChatOpenAI()

    chain = model.bind(functions=router.functions) | router

    assert router.functions == snapshot

    assert chain.invoke("Something about turtles?") == "Accepted draft: turtles!"

    revise.assert_not_called()
    accept.assert_called_once_with({"draft": "turtles"})
