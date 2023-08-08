"""Fake LLM wrapper for testing purposes."""
from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class EchoLLM(LLM):
    """Echo LLM wrapper for testing purposes.
    The LLM will return the prompt as the response.
    """

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "echo"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return prompt
