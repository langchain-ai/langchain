from __future__ import annotations

from langchain.automaton.open_ai_functions import OpenAIFunctionsRouter
from langchain.automaton.tests.utils import FakeChatOpenAI
from langchain.schema import AIMessage
from langchain.schema.runnable import RunnableLambda


def test_openai_functions_router() -> None:
    """Test the OpenAIFunctionsRouter."""

    def revise(notes: str) -> str:
        """Revises the draft."""
        return f"Revised draft: {notes}!"

    def accept(draft: str) -> str:
        """Accepts the draft."""
        return f"Accepted draft: {draft}!"

    router = OpenAIFunctionsRouter(
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
        runnables={
            "revise": RunnableLambda(lambda x: revise(x["revise"])),
            "accept": RunnableLambda(lambda x: accept(x["draft"])),
        },
    )

    model = FakeChatOpenAI(
        message_iter=iter(
            [
                AIMessage(
                    content="",
                    additional_kwargs={
                        "function_call": {
                            "name": "accept",
                            "arguments": '{\n  "draft": "turtles"\n}',
                        }
                    },
                )
            ]
        )
    )

    chain = model.bind(functions=router.functions) | router

    assert chain.invoke("Something about turtles?") == "Accepted draft: turtles!"
