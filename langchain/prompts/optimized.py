"""Optimized prompt schema definition."""
import re
from typing import Any, Callable, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.embeddings.base import Embeddings
from langchain.prompts.base import DEFAULT_FORMATTER_MAPPING
from langchain.prompts.dynamic import DynamicPrompt


class OptimizedPrompt(BaseModel, DynamicPrompt):
    r"""Schema to represent an optimized prompt for an LLM.

    Example:
        .. code-block:: python

            from langchain import DynamicPrompt
            optimized_prompt = OptimizedPrompt(
                examples=["Say hi. Hi", "Say ho. Ho"],
                example_separator="\n\n",
                prefix="",
                suffix="\n\nSay {foo}"
                input_variables=["foo"],
                max_length=200,
                get_text_length=word_count,
                embeddings_cls=OpenAIEmbeddings,
                vectorstore_cls=FAISS
            )
    """
    examples: List[str]
    """A list of the examples that the prompt template expects."""

    example_separator: str = "\n\n"
    """Example separator, e.g. \n\n, for the dynamic prompt creation."""

    input_variables: List[str] = []
    """A list of the names of the variables the prompt template expects."""

    prefix: str = ""
    """Prefix for the prompt."""

    suffix: str = ""
    """Suffix for the prompt."""

    template_format: str = "f-string"
    """The format of the prompt template. Options are: 'f-string'."""

    get_text_length: Callable[[str], int] = lambda x: len(re.split("\n| ", x))
    """Function to measure prompt length. Defaults to word count."""

    max_length: int = 2048
    """Max length for the prompt, beyond which examples are cut."""

    vectorstore_client: VectorStore
    """Vectorstore class to use for storing the embeddings."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def template(self, example_list: List[str], **kwargs: Any) -> str:
        """Return template given example list."""
        template = self.example_separator.join(
            [self.prefix, *example_list, self.suffix]
        )
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    def format(self, **kwargs: Any) -> str:
        """Dynamically format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """
        curr_examples = self.examples
        template = self.template(curr_examples, **kwargs)
        while self.get_text_length(template) > self.max_length and curr_examples:
            curr_examples = curr_examples[:-1]
            template = self.template(curr_examples, **kwargs)
        return template

    @classmethod
    def from_embeddings()
