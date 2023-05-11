from typing import Any, Callable

from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLanguageModel
from langchain.prompts.chat import (
    BaseMessagePromptTemplate,
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import BaseOutputParser, PromptValue
from langchain.wrappers.chat_model_facade import ChatModelFacade
from langchain.wrappers.llm_facade import LLMFacade

SYSTEM_PROMPT_TEMPLATE = "{prompt}"
CONTINUE_INCOMPLETE_PROMPT_TEMPLATE = """
{completion_start_trunc}
...
{completion_end_trunc}"""
MERGE_INCOMPLETE_RESPONSES_PROMPT_TEMPLATE = """
Please stitch the following inputs together into a single coherent response:

Trailing text from part 1: ...{prev_trailing}
Beginning text from part 2: {new_leading}...

Combined: """
CONTINUE_INCOMPLETE_PLEASE_CONTINUE_PROMPT_TEMPLATE = (
    "Sorry, your response was incomplete. Please finish your response:"
)

SYSTEM_PROMPT = SystemMessagePromptTemplate.from_template(template=SYSTEM_PROMPT_TEMPLATE)
CONTINUE_INCOMPLETE_PROMPT = HumanMessagePromptTemplate.from_template(
    template=CONTINUE_INCOMPLETE_PROMPT_TEMPLATE
)
MERGE_INCOMPLETE_RESPONSES_PROMPT = HumanMessagePromptTemplate.from_template(
    template=MERGE_INCOMPLETE_RESPONSES_PROMPT_TEMPLATE
)
CONTINUE_INCOMPLETE_PLEASE_CONTINUE_PROMPT = HumanMessagePromptTemplate.from_template(
    template=CONTINUE_INCOMPLETE_PLEASE_CONTINUE_PROMPT_TEMPLATE
)


class StitchedOutputParser(BaseOutputParser[str]):
    """Used to finish parsing an output that is incomplete."""

    completion_validator: Callable[[str], bool]
    chat_model: BaseChatModel
    continue_prompt: BaseMessagePromptTemplate
    merge_prompt: BaseMessagePromptTemplate
    max_steps: int
    continuation_keep_start_chars: int
    continuation_keep_end_chars: int
    stitch_chars: int

    @classmethod
    def from_llm(
        cls,
        completion_validator: Callable[[str], bool],
        llm: BaseLanguageModel | BaseChatModel,
        continue_prompt: BaseMessagePromptTemplate = CONTINUE_INCOMPLETE_PROMPT,
        merge_prompt: BaseMessagePromptTemplate = MERGE_INCOMPLETE_RESPONSES_PROMPT_TEMPLATE,
        continue_incomplete_pls_continue_prompt: BaseMessagePromptTemplate = CONTINUE_INCOMPLETE_PLEASE_CONTINUE_PROMPT,
        max_steps: int = 10,
        continuation_keep_start_chars: int = 500,
        continuation_keep_end_chars: int = 500,
        stitch_chars: int = 50,
    ):
        return cls(
            completion_validator=completion_validator,
            chat_model=ChatModelFacade.of(llm),
            continue_prompt=ChatMessagePromptTemplate.from_template(
                [
                    SYSTEM_PROMPT,
                    continue_prompt,
                    continue_incomplete_pls_continue_prompt,
                ]
            ),
            merge_prompt=merge_prompt,
            max_steps=max_steps,
            continuation_keep_start_chars=continuation_keep_start_chars,
            continuation_keep_end_chars=continuation_keep_end_chars,
            stitch_chars=stitch_chars,
        )

    def parse_with_prompt(self, completion: str, prompt: PromptValue) -> Any:
        while not self._is_complete(completion):
            # get the continuation
            completion_start_trunc = completion[: self.continuation_keep_start_chars]
            completion_end_trunc = completion[-self.continuation_keep_end_chars :]
            continuation = self.chat_model(
                self.continue_prompt.format(
                    prompt=prompt,
                    completion_start_trunc=completion_start_trunc,
                    completion_end_trunc=completion_end_trunc,
                )
            ).content
            # stitch the continuation to the completion
            prev_trailing = completion[-self.stitch_chars :]
            new_leading = continuation[: self.stitch_chars]
            stitch = self.chat_model(
                prompt=MERGE_INCOMPLETE_RESPONSES_PROMPT_TEMPLATE.format(
                    PREV_TRAILING=prev_trailing, NEW_LEADING=new_leading
                )
            )
            completion = (
                completion[: self.stitch_chars]
                + stitch
                + continuation[self.stitch_chars :]
            )

        return completion

    def parse(self, completion: str) -> str:
        raise NotImplementedError("Use `parse_with_prompt` instead.")

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    def _is_complete(self, output) -> True:
        padded_output = f"START HERE\n---\n{output}\n---\nEND HERE"
        for chunk in chunk(padded_output):
            if not self.completion_validator(chunk):
                return False
        return True
