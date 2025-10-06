from __future__ import annotations

from typing import Annotated, Any, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableSerializable
from pydantic import SkipValidation
from typing_extensions import TypedDict, override

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
    NAIVE_COMPLETION_RETRY_WITH_ERROR,
)

T = TypeVar("T")


class RetryOutputParserRetryChainInput(TypedDict):
    """Retry chain input for RetryOutputParser."""

    prompt: str
    completion: str


class RetryWithErrorOutputParserRetryChainInput(TypedDict):
    """Retry chain input for RetryWithErrorOutputParser."""

    prompt: str
    completion: str
    error: str


class RetryOutputParser(BaseOutputParser[T]):
    """Wrap a parser and try to fix parsing errors.

    Does this by passing the original prompt and the completion to another
    LLM, and telling it the completion did not satisfy criteria in the prompt.
    """

    parser: Annotated[BaseOutputParser[T], SkipValidation()]
    """The parser to use to parse the output."""
    # Should be an LLMChain but we want to avoid top-level imports from
    # langchain_classic.chains
    retry_chain: Annotated[
        RunnableSerializable[RetryOutputParserRetryChainInput, str] | Any,
        SkipValidation(),
    ]
    """The RunnableSerializable to use to retry the completion (Legacy: LLMChain)."""
    max_retries: int = 1
    """The maximum number of times to retry the parse."""
    legacy: bool = True
    """Whether to use the run or arun method of the retry_chain."""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        parser: BaseOutputParser[T],
        prompt: BasePromptTemplate = NAIVE_RETRY_PROMPT,
        max_retries: int = 1,
    ) -> RetryOutputParser[T]:
        """Create an RetryOutputParser from a language model and a parser.

        Args:
            llm: llm to use for fixing
            parser: parser to use for parsing
            prompt: prompt to use for fixing
            max_retries: Maximum number of retries to parse.

        Returns:
            RetryOutputParser
        """
        chain = prompt | llm | StrOutputParser()
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
            except OutputParserException:
                if retries == self.max_retries:
                    raise
                retries += 1
                if self.legacy and hasattr(self.retry_chain, "run"):
                    completion = self.retry_chain.run(
                        prompt=prompt_value.to_string(),
                        completion=completion,
                    )
                else:
                    completion = self.retry_chain.invoke(
                        {
                            "prompt": prompt_value.to_string(),
                            "completion": completion,
                        },
                    )

        msg = "Failed to parse"
        raise OutputParserException(msg)

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
                    raise
                retries += 1
                if self.legacy and hasattr(self.retry_chain, "arun"):
                    completion = await self.retry_chain.arun(
                        prompt=prompt_value.to_string(),
                        completion=completion,
                        error=repr(e),
                    )
                else:
                    completion = await self.retry_chain.ainvoke(
                        {
                            "prompt": prompt_value.to_string(),
                            "completion": completion,
                        },
                    )

        msg = "Failed to parse"
        raise OutputParserException(msg)

    @override
    def parse(self, completion: str) -> T:
        msg = "This OutputParser can only be called by the `parse_with_prompt` method."
        raise NotImplementedError(msg)

    @override
    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return "retry"

    @property
    @override
    def OutputType(self) -> type[T]:
        return self.parser.OutputType


class RetryWithErrorOutputParser(BaseOutputParser[T]):
    """Wrap a parser and try to fix parsing errors.

    Does this by passing the original prompt, the completion, AND the error
    that was raised to another language model and telling it that the completion
    did not work, and raised the given error. Differs from RetryOutputParser
    in that this implementation provides the error that was raised back to the
    LLM, which in theory should give it more information on how to fix it.
    """

    parser: Annotated[BaseOutputParser[T], SkipValidation()]
    """The parser to use to parse the output."""
    # Should be an LLMChain but we want to avoid top-level imports from
    # langchain_classic.chains
    retry_chain: Annotated[
        RunnableSerializable[RetryWithErrorOutputParserRetryChainInput, str] | Any,
        SkipValidation(),
    ]
    """The RunnableSerializable to use to retry the completion (Legacy: LLMChain)."""
    max_retries: int = 1
    """The maximum number of times to retry the parse."""
    legacy: bool = True
    """Whether to use the run or arun method of the retry_chain."""

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
        chain = prompt | llm | StrOutputParser()
        return cls(parser=parser, retry_chain=chain, max_retries=max_retries)

    @override
    def parse_with_prompt(self, completion: str, prompt_value: PromptValue) -> T:
        retries = 0

        while retries <= self.max_retries:
            try:
                return self.parser.parse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise
                retries += 1
                if self.legacy and hasattr(self.retry_chain, "run"):
                    completion = self.retry_chain.run(
                        prompt=prompt_value.to_string(),
                        completion=completion,
                        error=repr(e),
                    )
                else:
                    completion = self.retry_chain.invoke(
                        {
                            "completion": completion,
                            "prompt": prompt_value.to_string(),
                            "error": repr(e),
                        },
                    )

        msg = "Failed to parse"
        raise OutputParserException(msg)

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
                    raise
                retries += 1
                if self.legacy and hasattr(self.retry_chain, "arun"):
                    completion = await self.retry_chain.arun(
                        prompt=prompt_value.to_string(),
                        completion=completion,
                        error=repr(e),
                    )
                else:
                    completion = await self.retry_chain.ainvoke(
                        {
                            "prompt": prompt_value.to_string(),
                            "completion": completion,
                            "error": repr(e),
                        },
                    )

        msg = "Failed to parse"
        raise OutputParserException(msg)

    @override
    def parse(self, completion: str) -> T:
        msg = "This OutputParser can only be called by the `parse_with_prompt` method."
        raise NotImplementedError(msg)

    @override
    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return "retry_with_error"

    @property
    @override
    def OutputType(self) -> type[T]:
        return self.parser.OutputType
