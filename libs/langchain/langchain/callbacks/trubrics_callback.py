import os
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain.adapters.openai import convert_message_to_dict
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.schema.messages import BaseMessage


class TrubricsCallbackHandler(BaseCallbackHandler):
    """
    Callback handler for Trubrics.

    Args:
        project: a trubrics project, default project is "default"
        email: a trubrics account email, can equally be set in env variables
        password: a trubrics account password, can equally be set in env variables
        **kwargs: all other kwargs are parsed and set to trubrics prompt variables,
            or added to the `metadata` dict
    """

    def __init__(
        self,
        project: str = "default",
        email: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        try:
            from trubrics import Trubrics
        except ImportError:
            raise ImportError(
                "The TrubricsCallbackHandler requires installation of "
                "the trubrics package. "
                "Please install it with `pip install trubrics`."
            )

        self.trubrics = Trubrics(
            project=project,
            email=email or os.environ["TRUBRICS_EMAIL"],
            password=password or os.environ["TRUBRICS_PASSWORD"],
        )
        self.config_model: dict = {}
        self.prompt: Optional[str] = None
        self.messages: Optional[list] = None
        self.trubrics_kwargs: Optional[dict] = kwargs if kwargs else None

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        self.prompt = prompts[0]

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any
    ) -> None:
        self.messages = [convert_message_to_dict(message) for message in messages[0]]
        self.prompt = self.messages[-1]["content"]

    def on_llm_end(self, response: LLMResult, run_id: UUID, **kwargs: Any) -> None:
        tags = ["langchain"]
        user_id = None
        session_id = None
        metadata: dict = {"langchain_run_id": run_id}
        if self.messages:
            metadata["messages"] = self.messages
        if self.trubrics_kwargs:
            if self.trubrics_kwargs.get("tags"):
                tags.append(*self.trubrics_kwargs.pop("tags"))
            user_id = self.trubrics_kwargs.pop("user_id", None)
            session_id = self.trubrics_kwargs.pop("session_id", None)
            metadata.update(self.trubrics_kwargs)

        for generation in response.generations:
            self.trubrics.log_prompt(
                config_model={
                    "model": response.llm_output.get("model_name")
                    if response.llm_output
                    else "NA"
                },
                prompt=self.prompt,
                generation=generation[0].text,
                user_id=user_id,
                session_id=session_id,
                tags=tags,
                metadata=metadata,
            )
