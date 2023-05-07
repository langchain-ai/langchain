from __future__ import annotations

from typing import Callable, TypeVar

from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate, StringPromptValue
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (
    BaseOutputParser,
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
        return self.parser._type


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


class MultiAttemptRetryWithErrorOutputParser(RetryWithErrorOutputParser):
    """Wraps a parser and tries to fix parsing errors by making multiple attempts to parse the completion with a language model chain.

    This class provides the error that was raised back to the language model, which should give it more information on how to fix it. It also allows for an additional validator function to be provided, which can be used to further validate the parsed completion. The class overrides the `parse_with_prompt` method to handle multiple attempts at parsing the completion with the language model chain. The `get_format_instructions` method returns the format instructions of the wrapped parser. This class can only be called by the `parse_with_prompt` method.
    """

    parser: BaseOutputParser[T]
    retry_chain: LLMChain
    attempts: int
    additional_validator: Callable[[str], None]

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        parser: BaseOutputParser[T],
        prompt: BasePromptTemplate = NAIVE_RETRY_WITH_ERROR_PROMPT,
        attempts: int = 3,
        additional_validator: Callable[[str], None] = None,
    ) -> RetryWithErrorOutputParser[T]:
        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            parser=parser,
            retry_chain=chain,
            attempts=attempts,
            additional_validator=additional_validator,
        )

    def parse_with_prompt(self, completion: str, prompt_value: PromptValue) -> T:
        try:
            parsed_completion = self.parser.parse(completion)
            self.additional_validator(new_completion)
        except OutputParserException as e:
            for _ in range(self.attempts):
                try:
                    new_completion = self.retry_chain.run(
                        prompt=prompt_value.to_string(),
                        completion=completion,
                        error=repr(e),
                    )
                    parsed_completion = self.parser.parse_with_prompt(
                        new_completion, StringPromptValue(text=prompt_value.to_string())
                    )
                    self.additional_validator(new_completion)
                    break
                except OutputParserException as e:
                    continue
            else:
                raise e

        return parsed_completion

    def parse(self, completion: str) -> T:
        raise NotImplementedError(
            "This OutputParser can only be called by the `parse_with_prompt` method."
        )

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()
