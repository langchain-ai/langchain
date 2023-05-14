from __future__ import annotations

from typing import Callable, TypeVar

from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.prompts.base import BasePromptTemplate, StringPromptValue
from langchain.prompts.chat import ChatPromptValue
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (
    AIMessage,
    BaseMessage,
    BaseOutputParser,
    HumanMessage,
    OutputParserException,
    PromptValue,
)

NAIVE_COMPLETION_RETRY = """Prompt:
{prompt}
Completion:
{completion}

Above, the Completion did not satisfy the constraints given in the Prompt.
Please try again:"""

NAIVE_COMPLETION_RETRY_WITH_ERROR = """Prompt:
{prompt}
Completion:
{completion}

Above, the Completion did not satisfy the constraints given in the Prompt.
Details: {error}
Please try again:"""

NAIVE_RETRY_PROMPT = PromptTemplate.from_template(NAIVE_COMPLETION_RETRY)
NAIVE_RETRY_WITH_ERROR_PROMPT = PromptTemplate.from_template(
    NAIVE_COMPLETION_RETRY_WITH_ERROR
)

T = TypeVar("T")


class RetryOutputParser(BaseOutputParser[T]):
    """Wraps a parser and tries to fix parsing errors.

    Does this by passing the original prompt and the completion to another
    LLM, and telling it the completion did not satisfy criteria in the prompt.
    """

    parser: BaseOutputParser[T]
    retry_chain: LLMChain

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        parser: BaseOutputParser[T],
        prompt: BasePromptTemplate = NAIVE_RETRY_PROMPT,
    ) -> RetryOutputParser[T]:
        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(parser=parser, retry_chain=chain)

    def parse_with_prompt(self, completion: str, prompt_value: PromptValue) -> T:
        try:
            parsed_completion = self.parser.parse(completion)
        except OutputParserException:
            new_completion = self.retry_chain.run(
                prompt=prompt_value.to_string(), completion=completion
            )
            parsed_completion = self.parser.parse(new_completion)

        return parsed_completion

    def parse(self, completion: str) -> T:
        raise NotImplementedError(
            "This OutputParser can only be called by the `parse_with_prompt` method."
        )

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return "retry"


class RetryWithErrorOutputParser(BaseOutputParser[T]):
    """Wraps a parser and tries to fix parsing errors.

    Does this by passing the original prompt, the completion, AND the error
    that was raised to another language and telling it that the completion
    did not work, and raised the given error. Differs from RetryOutputParser
    in that this implementation provides the error that was raised back to the
    LLM, which in theory should give it more information on how to fix it.
    """

    parser: BaseOutputParser[T]
    retry_chain: LLMChain

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        parser: BaseOutputParser[T],
        prompt: BasePromptTemplate = NAIVE_RETRY_WITH_ERROR_PROMPT,
    ) -> RetryWithErrorOutputParser[T]:
        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(parser=parser, retry_chain=chain)

    def parse_with_prompt(self, completion: str, prompt_value: PromptValue) -> T:
        try:
            parsed_completion = self.parser.parse(completion)
        except OutputParserException as e:
            new_completion = self.retry_chain.run(
                prompt=prompt_value.to_string(), completion=completion, error=repr(e)
            )
            parsed_completion = self.parser.parse(new_completion)

        return parsed_completion

    def parse(self, completion: str) -> T:
        raise NotImplementedError(
            "This OutputParser can only be called by the `parse_with_prompt` method."
        )

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return "retry_with_error"


class MultiAttemptRetryWithErrorOutputParser(BaseOutputParser[T]):
    """Wraps a parser and tries to fix parsing errors by making multiple attempts to parse the completion with a language model chain.

    This class provides the error that was raised back to the language model, which should give it more information on how to fix it. It also allows for an additional validator function to be provided, which can be used to further validate the parsed completion. The class overrides the `parse_with_prompt` method to handle multiple attempts at parsing the completion with the language model chain. The `get_format_instructions` method returns the format instructions of the wrapped parser. This class can only be called by the `parse_with_prompt` method.
    """

    llm: BaseLanguageModel | BaseChatModel
    parser: BaseOutputParser[T]
    retry_prompt: BasePromptTemplate
    attempts: int
    additional_validator: Callable[[str], None] | None

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        parser: BaseOutputParser[T],
        retry_prompt: BasePromptTemplate = NAIVE_RETRY_WITH_ERROR_PROMPT,
        attempts: int = 3,
        additional_validator: Callable[[str], None] = None,
    ) -> RetryWithErrorOutputParser[T]:
        return cls(
            llm=llm,
            parser=parser,
            retry_prompt=retry_prompt,
            attempts=attempts,
            additional_validator=additional_validator,
        )

    def parse_with_prompt(self, completion: str, prompt_value: PromptValue) -> T:
        print("parse_with_prompt:1")
        print("prompt_value: ", prompt_value)
        try:
            print(
                "MultiAttemptRetryWithErrorOutputParser:parse_with_prompt:1.1: completion: ",
                completion,
            )
            parsed_completion = self.parser.parse(completion)
            print(
                "MultiAttemptRetryWithErrorOutputParser:parse_with_prompt:1.2: parsed_completion: ",
                parsed_completion,
            )
            print("parse_with_prompt:2")
            if self.additional_validator is not None:
                self.additional_validator(parsed_completion)
                print("parse_with_prompt:3")
        except OutputParserException as e:
            print("parse_with_prompt:4")
            context: list[BaseMessage] | str
            if isinstance(prompt_value, StringPromptValue):
                print("parse_with_prompt:5")
                context = prompt_value.to_string()
            elif isinstance(prompt_value, ChatPromptValue):
                print("parse_with_prompt:6")
                context = prompt_value.to_messages()
            else:
                print("parse_with_prompt:7")
                raise ValueError(
                    f"Prompt value must be a string or a list of BaseMessages, but got {prompt_value}."
                )

            for _ in range(self.attempts):
                print("parse_with_prompt:8")
                feedback = self.retry_prompt.format(
                    prompt=prompt_value.to_string(),
                    completion=completion,
                    error=repr(e),
                )
                print("parse_with_prompt:9")
                if isinstance(context, str):
                    print("parse_with_prompt:10")
                    context += f"\n\n{completion}\n\n{feedback}\n\n"
                elif isinstance(context, list) and all(
                    isinstance(x, BaseMessage) for x in context
                ):
                    print("parse_with_prompt:11")
                    context += [
                        AIMessage(content=completion),
                        HumanMessage(content=feedback),
                    ]
                else:
                    print("parse_with_prompt:12")
                    raise ValueError(
                        f"`context` must be a string or a list of BaseMessages, but is {context}."
                    )
                try:
                    print("parse_with_prompt:13")
                    completion = self.llm(context)
                    print("parse_with_prompt:14")
                    if isinstance(completion, str):
                        print("parse_with_prompt:15")
                        pass
                    elif isinstance(completion, BaseMessage):
                        print("parse_with_prompt:16")
                        completion = completion.content
                    else:
                        print("parse_with_prompt:17")
                        raise ValueError(
                            f"LLM must return a string or a BaseMessage, but got {completion}."
                        )
                    print("parse_with_prompt:18")
                    parsed_completion = self.parser.parse_with_prompt(
                        completion, prompt_value
                    )
                    print("parse_with_prompt:19")
                    if self.additional_validator is not None:
                        self.additional_validator(completion)
                        print("parse_with_prompt:20")
                    break
                except OutputParserException as e:
                    print("parse_with_prompt:21")
                    continue
            else:
                print("parse_with_prompt:22")
                raise e

        print("parse_with_prompt:23")
        return parsed_completion

    def parse(self, completion: str) -> T:
        raise NotImplementedError(
            "This OutputParser can only be called by the `parse_with_prompt` method."
        )

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return "multi_attempt_retry_with_error"
