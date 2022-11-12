"""Optimized prompt schema definition."""
import re
from typing import Any, Callable, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.embeddings.base import Embeddings
from langchain.prompts.base import DEFAULT_FORMATTER_MAPPING
from langchain.vectorstores.base import VectorStore


class OptimizedPrompt(BaseModel):
    r"""Schema to represent an optimized prompt for an LLM.

    Example:
        .. code-block:: python

            from langchain import DynamicPrompt
            vectorstore = FAISS.from_texts(examples, OpenAIEmbeddings()
            optimized_prompt = OptimizedPrompt(
                example_separator="\n\n",
                prefix="",
                suffix="\n\nSay {foo}"
                input_variables=["foo"],
                max_length=200,
                get_text_length=word_count,
                vectorstore=vectorstore)
            )
    """

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

    vectorstore: VectorStore
    """Vectorstore to use for storing the embeddings."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

        extra = Extra.forbid

    def template(self, example_list: List[str], **kwargs: Any) -> str:
        """Return template given full example list."""
        template = self.example_separator.join(
            [self.prefix, *example_list, self.suffix]
        )
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    def format(self, k: int = 4, **kwargs: Any) -> str:
        """Optimize the examples in the prompt for the given inputs.

        Args:
            k: Number of examples to aim for (may be trimmed by optimizer afterwards)
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """
        query = " ".join([v for k, v in kwargs.items()])
        example_docs = self.vectorstore.similarity_search(query, k=k)
        curr_examples = [str(e.page_content) for e in example_docs]
        template = self.template(curr_examples, **kwargs)
        while self.get_text_length(template) > self.max_length and curr_examples:
            curr_examples = curr_examples[:-1]
            template = self.template(curr_examples, **kwargs)
        return template

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that prefix, suffix and input variables are consistent."""
        input_variables = values["input_variables"]
        if len(input_variables) > 1:
            raise ValueError("Only one input variable allowed for optimized prompt;")
        prefix = values["prefix"]
        suffix = values["suffix"]
        template_format = values["template_format"]
        if template_format not in DEFAULT_FORMATTER_MAPPING:
            valid_formats = list(DEFAULT_FORMATTER_MAPPING)
            raise ValueError(
                f"Invalid template format. Got `{template_format}`;"
                f" should be one of {valid_formats}"
            )
        try:
            result = values["get_text_length"]("foo")
            assert isinstance(result, int)
        except AssertionError:
            raise ValueError(
                "Invalid text length callable, must take string & return int;"
            )
        dummy_inputs = {input_variable: "foo" for input_variable in input_variables}
        try:
            formatter_func = DEFAULT_FORMATTER_MAPPING[template_format]
            formatter_func(prefix + suffix, **dummy_inputs)
        except KeyError:
            raise ValueError("Invalid prompt schema.")
        return values

    @classmethod
    def from_examples(
        cls,
        examples: List[str],
        suffix: str,
        input_variables: List[str],
        embeddings: Embeddings,
        vectorstore_cls: VectorStore,
        example_separator: str = "\n\n",
        prefix: str = "",
        **vectorstore_cls_kwargs: Any,
    ) -> "OptimizedPrompt":
        """Create k-shot prompt optimizer using example list and embeddings.

        Reshuffles examples for the prompt dynamically based on query similarity.

        Args:
            examples: List of examples to use in the prompt.
            suffix: String to go after the list of examples. Should generally
                set up the user's input.
            input_variables: A list of variable names the final prompt template
                will expect.
            embeddings: An iniialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            example_separator: The seperator to use in between examples. Defaults
                to two new line characters.
            prefix: String that should go before any examples. Generally includes
                examples. Default to an empty string.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            The OptimizedPrompt instantiated, backed by a vector store.
        """
        vectorstore = vectorstore_cls.from_texts(
            examples, embeddings, **vectorstore_cls_kwargs
        )
        return cls(
            suffix=suffix,
            input_variables=input_variables,
            example_separator=example_separator,
            prefix=prefix,
            vectorstore=vectorstore,
        )
