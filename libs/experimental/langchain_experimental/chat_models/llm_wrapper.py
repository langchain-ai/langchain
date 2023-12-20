"""Generic Wrapper for chat LLMs, with sample implementations
for Llama-2-chat, Llama-2-instruct and Vicuna models.
"""
from typing import Any, List, Optional, cast

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import LLM
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
    HumanMessage,
    LLMResult,
    SystemMessage,
)

DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""  # noqa: E501


class ChatWrapper(BaseChatModel):
    llm: LLM
    sys_beg: str
    sys_end: str
    ai_n_beg: str
    ai_n_end: str
    usr_n_beg: str
    usr_n_end: str
    usr_0_beg: Optional[str] = None
    usr_0_end: Optional[str] = None

    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)
        llm_result = self.llm._generate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)
        llm_result = await self.llm._agenerate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    def _to_chat_prompt(
        self,
        messages: List[BaseMessage],
    ) -> str:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        if not messages:
            raise ValueError("at least one HumanMessage must be provided")

        if not isinstance(messages[0], SystemMessage):
            messages = [self.system_message] + messages

        if not isinstance(messages[1], HumanMessage):
            raise ValueError(
                "messages list must start with a SystemMessage or UserMessage"
            )

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("last message must be a HumanMessage")

        prompt_parts = []

        if self.usr_0_beg is None:
            self.usr_0_beg = self.usr_n_beg

        if self.usr_0_end is None:
            self.usr_0_end = self.usr_n_end

        prompt_parts.append(
            self.sys_beg + cast(str, messages[0].content) + self.sys_end
        )
        prompt_parts.append(
            self.usr_0_beg + cast(str, messages[1].content) + self.usr_0_end
        )

        for ai_message, human_message in zip(messages[2::2], messages[3::2]):
            if not isinstance(ai_message, AIMessage) or not isinstance(
                human_message, HumanMessage
            ):
                raise ValueError(
                    "messages must be alternating human- and ai-messages, "
                    "optionally prepended by a system message"
                )

            prompt_parts.append(
                self.ai_n_beg + cast(str, ai_message.content) + self.ai_n_end
            )
            prompt_parts.append(
                self.usr_n_beg + cast(str, human_message.content) + self.usr_n_end
            )

        return "".join(prompt_parts)

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        chat_generations = []

        for g in llm_result.generations[0]:
            chat_generation = ChatGeneration(
                message=AIMessage(content=g.text), generation_info=g.generation_info
            )
            chat_generations.append(chat_generation)

        return ChatResult(
            generations=chat_generations, llm_output=llm_result.llm_output
        )


class Llama2Chat(ChatWrapper):
    @property
    def _llm_type(self) -> str:
        return "llama-2-chat"

    sys_beg: str = "<s>[INST] <<SYS>>\n"
    sys_end: str = "\n<</SYS>>\n\n"
    ai_n_beg: str = " "
    ai_n_end: str = " </s>"
    usr_n_beg: str = "<s>[INST] "
    usr_n_end: str = " [/INST]"
    usr_0_beg: str = ""
    usr_0_end: str = " [/INST]"


class Orca(ChatWrapper):
    @property
    def _llm_type(self) -> str:
        return "orca-style"

    sys_beg: str = "### System:\n"
    sys_end: str = "\n\n"
    ai_n_beg: str = "### Assistant:\n"
    ai_n_end: str = "\n\n"
    usr_n_beg: str = "### User:\n"
    usr_n_end: str = "\n\n"


class Vicuna(ChatWrapper):
    @property
    def _llm_type(self) -> str:
        return "vicuna-style"

    sys_beg: str = ""
    sys_end: str = " "
    ai_n_beg: str = "ASSISTANT: "
    ai_n_end: str = " </s>"
    usr_n_beg: str = "USER: "
    usr_n_end: str = " "
