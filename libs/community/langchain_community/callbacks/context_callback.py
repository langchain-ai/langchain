"""Callback handler for Context AI"""
import os
from typing import Any, Dict, List
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


def import_context() -> Any:
    """Import the `getcontext` package."""
    try:
        import getcontext
        from getcontext.generated.models import (
            Conversation,
            Message,
            MessageRole,
            Rating,
        )
        from getcontext.token import Credential
    except ImportError:
        raise ImportError(
            "To use the context callback manager you need to have the "
            "`getcontext` python package installed (version >=0.3.0). "
            "Please install it with `pip install --upgrade python-context`"
        )
    return getcontext, Credential, Conversation, Message, MessageRole, Rating


class ContextCallbackHandler(BaseCallbackHandler):
    """Callback Handler that records transcripts to the Context service.

     (https://context.ai).

    Keyword Args:
        token (optional): The token with which to authenticate requests to Context.
            Visit https://with.context.ai/settings to generate a token.
            If not provided, the value of the `CONTEXT_TOKEN` environment
            variable will be used.

    Raises:
        ImportError: if the `context-python` package is not installed.

    Chat Example:
        >>> from langchain_community.llms import ChatOpenAI
        >>> from langchain_community.callbacks import ContextCallbackHandler
        >>> context_callback = ContextCallbackHandler(
        ...     token="<CONTEXT_TOKEN_HERE>",
        ... )
        >>> chat = ChatOpenAI(
        ...     temperature=0,
        ...     headers={"user_id": "123"},
        ...     callbacks=[context_callback],
        ...     openai_api_key="API_KEY_HERE",
        ... )
        >>> messages = [
        ...     SystemMessage(content="You translate English to French."),
        ...     HumanMessage(content="I love programming with LangChain."),
        ... ]
        >>> chat.invoke(messages)

    Chain Example:
        >>> from langchain.chains import LLMChain
        >>> from langchain_community.chat_models import ChatOpenAI
        >>> from langchain_community.callbacks import ContextCallbackHandler
        >>> context_callback = ContextCallbackHandler(
        ...     token="<CONTEXT_TOKEN_HERE>",
        ... )
        >>> human_message_prompt = HumanMessagePromptTemplate(
        ...     prompt=PromptTemplate(
        ...         template="What is a good name for a company that makes {product}?",
        ...         input_variables=["product"],
        ...    ),
        ... )
        >>> chat_prompt_template = ChatPromptTemplate.from_messages(
        ...   [human_message_prompt]
        ... )
        >>> callback = ContextCallbackHandler(token)
        >>> # Note: the same callback object must be shared between the
        ...   LLM and the chain.
        >>> chat = ChatOpenAI(temperature=0.9, callbacks=[callback])
        >>> chain = LLMChain(
        ...   llm=chat,
        ...   prompt=chat_prompt_template,
        ...   callbacks=[callback]
        ... )
        >>> chain.run("colorful socks")
    """

    def __init__(self, token: str = "", verbose: bool = False, **kwargs: Any) -> None:
        (
            self.context,
            self.credential,
            self.conversation_model,
            self.message_model,
            self.message_role_model,
            self.rating_model,
        ) = import_context()

        token = token or os.environ.get("CONTEXT_TOKEN") or ""

        self.client = self.context.ContextAPI(credential=self.credential(token))

        self.chain_run_id = None

        self.llm_model = None

        self.messages: List[Any] = []
        self.metadata: Dict[str, str] = {}

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Any:
        """Run when the chat model is started."""
        llm_model = kwargs.get("invocation_params", {}).get("model", None)
        if llm_model is not None:
            self.metadata["model"] = llm_model

        if len(messages) == 0:
            return

        for message in messages[0]:
            role = self.message_role_model.SYSTEM
            if message.type == "human":
                role = self.message_role_model.USER
            elif message.type == "system":
                role = self.message_role_model.SYSTEM
            elif message.type == "ai":
                role = self.message_role_model.ASSISTANT

            self.messages.append(
                self.message_model(
                    message=message.content,
                    role=role,
                )
            )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends."""
        if len(response.generations) == 0 or len(response.generations[0]) == 0:
            return

        if not self.chain_run_id:
            generation = response.generations[0][0]
            self.messages.append(
                self.message_model(
                    message=generation.text,
                    role=self.message_role_model.ASSISTANT,
                )
            )

            self._log_conversation()

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts."""
        self.chain_run_id = kwargs.get("run_id", None)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends."""
        self.messages.append(
            self.message_model(
                message=outputs["text"],
                role=self.message_role_model.ASSISTANT,
            )
        )

        self._log_conversation()

        self.chain_run_id = None

    def _log_conversation(self) -> None:
        """Log the conversation to the context API."""
        if len(self.messages) == 0:
            return

        self.client.log.conversation_upsert(
            body={
                "conversation": self.conversation_model(
                    messages=self.messages,
                    metadata=self.metadata,
                )
            }
        )

        self.messages = []
        self.metadata = {}
