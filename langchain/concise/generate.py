from typing import Callable, NoReturn, TypeVar
from langchain.concise.config import get_default_model, get_default_text_splitter
from langchain.prompts.chat import ChatPromptValue

from pydantic import BaseModel

from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLanguageModel
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.output_parsers.remove_quotes import RemoveQuotesOutputParser
from langchain.output_parsers.retry import MultiAttemptRetryWithErrorOutputParser
from langchain.prompts.base import StringPromptValue
from langchain.schema import BaseMessage, HumanMessage
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
        else:
            raise ValueError(
                f"Expected llm to be a BaseChatModel or BaseLanguageModel. Got {type(llm)}."
            )
    elif isinstance(input, str):
        if isinstance(llm, BaseChatModel):
            output = LLMFacade.of(llm)(input, stop=stop)
        elif isinstance(llm, BaseLanguageModel):
            output = llm(input, stop=stop)
        else:
            raise ValueError(
                f"Expected llm to be a BaseChatModel or BaseLanguageModel. Got {type(llm)}."
            )
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
    type=None,
    llm=None,
    stop=None,
    attempts=3,
    additional_validator: Callable[[str], None] = None,
    remove_quotes=False,
) -> T:
    llm = llm or get_default_model()

    if type is not None:
        if issubclass(type, BaseModel):
            parser = PydanticOutputParser(pydantic_object=type)
        elif issubclass(type, str):

            class StringParser(BaseOutputParser[str]):
                def parse(self, output: str) -> str:
                    return output

            parser = StringParser()
        elif issubclass(type, bool):
            parser = BooleanOutputParser()
        elif issubclass(type, float):

            class FloatParser(BaseOutputParser[float]):
                def parse(self, output: str) -> float:
                    for word in output.split():
                        try:
                            return float(word)
                        except ValueError:
                            pass
                    else:
                        raise ValueError(
                            f"Could not find a float in the output: {output}"
                        )

            parser = FloatParser()
        elif issubclass(type, int):

            class IntParser(BaseOutputParser[int]):
                def parse(self, output: str) -> int:
                    for word in output.split():
                        try:
                            return int(word)
                        except ValueError:
                            pass
                    else:
                        raise ValueError(
                            f"Could not find an integer in the output: {output}"
                        )

            parser = IntParser()
        else:
            raise ValueError(
                f"Invalid type: {type}. Must be a str, bool, int, float, or BaseModel."
            )
    if parser is None:
        return _generate(input, llm=llm, stop=stop)
    if remove_quotes:
        parser = RemoveQuotesOutputParser(parser)

    try:
        if isinstance(input, list):
            input.append(
                HumanMessage(
                    text=f"Formatting directions: {parser.get_format_instructions()}"
                )
            )
        elif isinstance(input, str):
            input += f"\n\n## Formatting directions\n\n{parser.get_format_instructions()}\n\n## Output\n\n"
        else:
            raise ValueError(
                f"Invalid input type: {type(input)}. Must be a string or list of messages."
            )
    except (NotImplementedError, AttributeError):
        pass

    print(123123)
    retry_with_error_parser = MultiAttemptRetryWithErrorOutputParser.from_llm(
        parser=parser,
        llm=llm,
        attempts=attempts,
        additional_validator=additional_validator,
    )
    print(456456)
    response = _generate(input, llm=llm, stop=stop)
    print(789789)
    return retry_with_error_parser.parse_with_prompt(
        response,
        prompt_value=StringPromptValue(text=input)
        if isinstance(input, str)
        else ChatPromptValue(messages=input)
        if isinstance(input, list)
        else None,
    )
