from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeChatLLMT(BaseChatModel):
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


openapi_endpoint_doc_mock = {
    "summary": "Test Action",
    "operationId": "test_action",
    "description": "Greets an employee",
    "responses": {
        "description": "Successful Response",
        "content": {
            "application/json": {
                "schema": {
                    "type": "string",
                    "title": "Response Test Action",
                    "description": "The greeting",
                }
            }
        },
    },
    "requestBody": {
        "content": {
            "application/json": {
                "schema": {
                    "properties": {
                        "name": {
                            "type": "string",
                            "title": "Name",
                            "description": "Name of the employee",
                        }
                    },
                    "type": "object",
                    "required": ["name"],
                    "title": "TestActionInput",
                }
            }
        },
        "required": True,
    },
}
