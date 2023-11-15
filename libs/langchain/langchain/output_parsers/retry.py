from __future__ import annotations

from typing import Any, TypeVar

from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (
    BaseOutputParser,
    BasePromptTemplate,
    OutputParserException,
    PromptValue,
)
from langchain.schema.language_model import BaseLanguageModel

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
    """The parser to use to parse the output."""
    # Should be an LLMChain but we want to avoid top-level imports from langchain.chains
    retry_chain: Any
    """The LLMChain to use to retry the completion."""
    max_retries: int = 1
    """The maximum number of times to retry the parse."""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        parser: BaseOutputParser[T],
        prompt: BasePromptTemplate = NAIVE_RETRY_PROMPT,
        max_retries: int = 1,
    ) -> RetryOutputParser[T]:
        """Create an OutputFixingParser from a language model and a parser.

        Args:
            llm: llm to use for fixing
            parser: parser to use for parsing
            prompt: prompt to use for fixing
            max_retries: Maximum number of retries to parse.

        Returns:
            RetryOutputParser
        """
        from langchain.chains.llm import LLMChain

        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(parser=parser, retry_chain=chain, max_retries=max_retries)

    def parse_with_prompt(self, completion: str, prompt_value: PromptValue) -> T:
        """Parse the output of an LLM call using a wrapped parser.

        Args:
            completion: The chain completion to parse.
            prompt_value: The prompt to use to parse the completion.

        Returns:
            The parsed completion.
        """
        retries = 0

        while retries <= self.max_retries:
            try:
                return self.parser.parse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise e
                else:
                    retries += 1
                    completion = self.retry_chain.run(
                        prompt=prompt_value.to_string(), completion=completion
                    )

        raise OutputParserException("Failed to parse")

    async def aparse_with_prompt(self, completion: str, prompt_value: PromptValue) -> T:
        """Parse the output of an LLM call using a wrapped parser.

        Args:
            completion: The chain completion to parse.
            prompt_value: The prompt to use to parse the completion.

        Returns:
            The parsed completion.
        """
        retries = 0

        while retries <= self.max_retries:
            try:
                return await self.parser.aparse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise e
                else:
                    retries += 1
                    completion = await self.retry_chain.arun(
                        prompt=prompt_value.to_string(), completion=completion
                    )

        raise OutputParserException("Failed to parse")

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
    that was raised to another language model and telling it that the completion
    did not work, and raised the given error. Differs from RetryOutputParser
    in that this implementation provides the error that was raised back to the
    LLM, which in theory should give it more information on how to fix it.
    """

    parser: BaseOutputParser[T]
    """The parser to use to parse the output."""
    # Should be an LLMChain but we want to avoid top-level imports from langchain.chains
    retry_chain: Any
    """The LLMChain to use to retry the completion."""
    max_retries: int = 1
    """The maximum number of times to retry the parse."""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        parser: BaseOutputParser[T],
        prompt: BasePromptTemplate = NAIVE_RETRY_WITH_ERROR_PROMPT,
        max_retries: int = 1,
    ) -> RetryWithErrorOutputParser[T]:
        """Create a RetryWithErrorOutputParser from an LLM.

        Args:
            llm: The LLM to use to retry the completion.
            parser: The parser to use to parse the output.
            prompt: The prompt to use to retry the completion.
            max_retries: The maximum number of times to retry the completion.

        Returns:
            A RetryWithErrorOutputParser.
        """
        from langchain.chains.llm import LLMChain

        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(parser=parser, retry_chain=chain, max_retries=max_retries)

    def parse_with_prompt(self, completion: str, prompt_value: PromptValue) -> T:
        retries = 0

        while retries <= self.max_retries:
            try:
                return self.parser.parse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise e
                else:
                    retries += 1
                    completion = self.retry_chain.run(
                        prompt=prompt_value.to_string(),
                        completion=completion,
                        error=repr(e),
                    )

        raise OutputParserException("Failed to parse")

    async def aparse_with_prompt(self, completion: str, prompt_value: PromptValue) -> T:
        retries = 0

        while retries <= self.max_retries:
            try:
                return await self.parser.aparse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise e
                else:
                    retries += 1
                    completion = await self.retry_chain.arun(
                        prompt=prompt_value.to_string(),
                        completion=completion,
                        error=repr(e),
                    )

        raise OutputParserException("Failed to parse")

    def parse(self, completion: str) -> T:
        raise NotImplementedError(
            "This OutputParser can only be called by the `parse_with_prompt` method."
        )

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return "retry_with_error"
