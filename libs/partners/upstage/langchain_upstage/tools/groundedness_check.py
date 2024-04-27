import os
from typing import Any, List, Literal, Optional, Type, Union

from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr
from langchain_core.tools import BaseTool
from langchain_core.utils import convert_to_secret_str

from langchain_upstage import ChatUpstage


class UpstageGroundednessCheckInput(BaseModel):
    """Input for the Groundedness Check tool."""

    context: Union[str, List[Document]] = Field(
        description="context in which the answer should be verified"
    )
    answer: str = Field(
        description="assistant's reply or a text that is subject to groundedness check"
    )


class UpstageGroundednessCheck(BaseTool):
    """Tool that checks the groundedness of a context and an assistant message.

    To use, you should have the environment variable `UPSTAGE_API_KEY`
    set with your API key or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

                from langchain_upstage import UpstageGroundednessCheck

                tool = UpstageGroundednessCheck()
    """

    name: str = "groundedness_check"
    description: str = (
        "A tool that checks the groundedness of an assistant response "
        "to user-provided context. UpstageGroundednessCheck ensures that "
        "the assistant’s response is not only relevant but also "
        "precisely aligned with the user's initial context, "
        "promoting a more reliable and context-aware interaction. "
        "When using retrieval-augmented generation (RAG), "
        "the Groundedness Check can be used to determine whether "
        "the assistant's message is grounded in the provided context."
    )
    upstage_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    api_wrapper: ChatUpstage

    args_schema: Type[BaseModel] = UpstageGroundednessCheckInput

    def __init__(self, **kwargs: Any) -> None:
        upstage_api_key = kwargs.get("upstage_api_key", None)
        if not upstage_api_key:
            upstage_api_key = kwargs.get("api_key", None)
        if not upstage_api_key:
            upstage_api_key = SecretStr(os.getenv("UPSTAGE_API_KEY", ""))
        upstage_api_key = convert_to_secret_str(upstage_api_key)

        if (
            not upstage_api_key
            or not upstage_api_key.get_secret_value()
            or upstage_api_key.get_secret_value() == ""
        ):
            raise ValueError("UPSTAGE_API_KEY must be set or passed")

        api_wrapper = ChatUpstage(
            model_name="solar-1-mini-answer-verification",
            upstage_api_key=upstage_api_key.get_secret_value(),
        )
        super().__init__(upstage_api_key=upstage_api_key, api_wrapper=api_wrapper)

    def formatDocumentsAsString(self, docs: List[Document]) -> str:
        return "\n".join([doc.page_content for doc in docs])

    def _run(
        self,
        context: Union[str, List[Document]],
        answer: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[str, Literal["grounded", "notGrounded", "notSure"]]:
        """Use the tool."""
        if isinstance(context, List):
            context = self.formatDocumentsAsString(context)
        response = self.api_wrapper.invoke(
            [HumanMessage(context), AIMessage(answer)], stream=False
        )
        return str(response.content)

    async def _arun(
        self,
        context: Union[str, List[Document]],
        answer: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[str, Literal["grounded", "notGrounded", "notSure"]]:
        if isinstance(context, List):
            context = self.formatDocumentsAsString(context)
        response = await self.api_wrapper.ainvoke(
            [HumanMessage(context), AIMessage(answer)], stream=False
        )
        return str(response.content)


@deprecated(
    since="0.1.3",
    removal="0.2.0",
    alternative_import="langchain_upstage.UpstageGroundednessCheck",
)
class GroundednessCheck(UpstageGroundednessCheck):
    pass
