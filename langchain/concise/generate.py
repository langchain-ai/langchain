from typing import Callable, NoReturn, TypeVar

from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLanguageModel
from langchain.output_parsers.remove_quotes import RemoveQuotesOutputParser
from langchain.output_parsers.retry import MultiAttemptRetryWithErrorOutputParser
from langchain.prompts.base import StringPromptValue
from langchain.schema import BaseMessage
from langchain.wrappers.chat_model_facade import ChatModelFacade
from langchain.wrappers.llm_facade import LLMFacade


def _generate(
    input: str | list[BaseMessage],
    llm: BaseChatModel | BaseLanguageModel = None,
    stop: str | list[str] = None,
) -> str:
    # handle all 4 combinations of chat and regular llms
    if isinstance(input, list) and all(isinstance(msg, BaseMessage) for msg in input):
        if isinstance(llm, BaseChatModel):
            output = llm(input, stop=stop).content
        elif isinstance(llm, BaseLanguageModel):
            output = ChatModelFacade.of(llm)(input, stop=stop)
    elif isinstance(input, str):
        if isinstance(llm, BaseChatModel):
            output = LLMFacade.of(llm)(input, stop=stop).content
        elif isinstance(llm, BaseLanguageModel):
            output = llm(input, stop=stop)
    else:
        raise ValueError(
            f"Invalid input type: {type(input)}. Must be a string or list of messages."
        )
    return output


T = TypeVar("T")
from langchain.schema import BaseOutputParser


def generate(
    input: str | list[BaseMessage],
    parser: BaseOutputParser[T] = None,
    llm=None,
    stop=None,
    attempts=None,
    additional_validator: Callable[[str], None] = None,
    remove_quotes=False,
) -> T:
    if parser is None:
        return _generate(input, llm=llm, stop=stop)
    if remove_quotes:
        parser = RemoveQuotesOutputParser(parser)
    retry_with_error_parser = MultiAttemptRetryWithErrorOutputParser.from_llm(
        parser=parser,
        llm=llm,
        attempts=attempts,
        additional_validator=additional_validator,
    )
    response = _generate(input, llm=llm, stop=stop)
    return retry_with_error_parser.parse_with_prompt(
        response, prompt_value=StringPromptValue(text=input)
    )
