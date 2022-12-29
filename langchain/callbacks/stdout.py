from typing import Any, Dict, List, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.input import print_text
from langchain.schema import AgentAction, LLMResult


class StdOutCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **extra: str
    ) -> None:
        print("Prompts after formatting:")
        for prompt in prompts:
            print_text(prompt, color="green", end="\n")

    def on_llm_end(self, response: LLMResult) -> None:
        pass

    def on_llm_error(self, error: Exception) -> None:
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **extra: str
    ) -> None:
        class_name = serialized["name"]
        print(f"\n\n\033[1m> Entering new {class_name} chain...\033[0m")

    def on_chain_end(self, outputs: Dict[str, Any]) -> None:
        print(f"\n\033[1m> Finished chain.\033[0m")
        if len(outputs) == 1:
            print(list(outputs.values())[0])

    def on_chain_error(self, error: Exception) -> None:
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        action: AgentAction,
        color: Optional[str] = None,
        **extra: str,
    ) -> None:
        print_text(action.log, color=color)

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        print_text(f"\n{observation_prefix}")
        print_text(output, color=color)
        print_text(f"\n{llm_prefix}")

    def on_tool_error(self, error: Exception) -> None:
        pass
