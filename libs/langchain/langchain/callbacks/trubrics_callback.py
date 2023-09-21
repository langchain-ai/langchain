import os
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.schema.messages import BaseMessage
from typing import Any, Dict, Optional, List

class TrubricsCallbackHandler(BaseCallbackHandler):
    def __init__(
            self,
            project: str,
            email: Optional[str] = None,
            password: Optional[str] = None,
        ) -> None:
        super().__init__()
        try:
            from trubrics import Trubrics
        except ImportError:
            raise ImportError(
                "The TrubricsCallbackHandler requires the trubrics package. "
                "Please install it with `pip install trubrics`."
            )

        self.trubrics = Trubrics(
            project=project,
            email=email or os.environ["TRUBRICS_EMAIL"],
            password=password or os.environ["TRUBRICS_PASSWORD"],
        )
        self.config_model = {}

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        return

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        return

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        # self.trubrics.log_prompt(
        #     config_model={"model": "gpt-3.5-turbo"},
        #     prompt="Tell me a joke",
        #     generation="Why did the chicken cross the road? To get to the other side.",
        # )
        return
