"""Prompt template that contains few shot examples."""
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Extra, root_validator

from langchain.prompts.base import (
    DEFAULT_FORMATTER_MAPPING,
    BasePromptTemplate,
    check_valid_template,
)
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    BaseChatPromptTemplate,
    BaseMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ExampleAIMessagePromptTemplate,
    ExampleHumanMessagePromptTemplate,
)
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.prompt import (
    BaseStringPromptTemplate,
    PromptTemplate,
    StringPromptTemplate,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)


class BaseFewShotPromptTemplate(BaseModel):
    """Base class for creating few-shot prompts."""

    suffix: str
    """A prompt template string to put after the examples."""

    prefix: str = ""
    """A prompt template string to put before the examples."""

    examples: Optional[List[dict]] = None
    """Examples to format into the prompt.
    Either this or example_selector should be provided."""

    example_selector: Optional[BaseExampleSelector] = None
    """ExampleSelector to choose the examples to format into the prompt.
    Either this or examples should be provided."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def _get_examples(self, **kwargs: Any) -> List[dict]:
        if self.examples is not None:
            return self.examples
        elif self.example_selector is not None:
            return self.example_selector.select_examples(kwargs)
        else:
            raise ValueError("No examples or example selector provided")

    def dict(self, **kwargs: Any) -> Dict:
        """Return a dictionary of the prompt."""
        if self.example_selector:
            raise ValueError("Saving an example selector is not currently supported")
        return super().dict(**kwargs)


class FewShotStringPromptTemplate(BaseStringPromptTemplate, BaseFewShotPromptTemplate):
    """Prompt template that contains few shot examples."""

    example_prompt: StringPromptTemplate
    """PromptTemplate used to format an individual example."""

    example_separator: str = "\n\n"
    """String separator used to join the prefix, the examples, and suffix."""

    @root_validator(pre=True)
    def check_examples_and_selector(cls, values: Dict) -> Dict:
        """Check that one and only one of examples/example_selector are provided."""
        examples = values.get("examples", None)
        example_selector = values.get("example_selector", None)
        if examples and example_selector:
            raise ValueError(
                "Only one of 'examples' and 'example_selector' should be provided"
            )

        if examples is None and example_selector is None:
            raise ValueError(
                "One of 'examples' and 'example_selector' should be provided"
            )

        return values

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that prefix, suffix and input variables are consistent."""
        if values["validate_template"]:
            check_valid_template(
                values["prefix"] + values["suffix"],
                values["template_format"],
                values["input_variables"] + list(values["partial_variables"]),
            )
        return values

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        # Get the examples to use.
        examples = self._get_examples(**kwargs)
        # Format the examples.
        example_strings = [
            self.example_prompt.format(**example) for example in examples
        ]
        # Create the overall template.
        pieces = [self.prefix, *example_strings, self.suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])

        # Format the template with the input variables.
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "few_shot"


# For backwards compatibility.
FewShotPromptTemplate = FewShotStringPromptTemplate


class FewShotChatPromptTemplate(BaseChatPromptTemplate, BaseFewShotPromptTemplate):
    """Prompt template that contains few shot examples."""

    example_input: str
    example_output: str
    examples: List[Dict[str, str]]

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "few_shot_chat"

    def format(self, **kwargs: Any) -> Sequence[BaseMessage]:
        """Format to a sequence of BaseMessages."""

        message_prompts = []
        if self.prefix:
            prefix_message_prompt = SystemMessagePromptTemplate(
                prompt=StringPromptTemplate.from_template(self.prefix)
            )
            message_prompts.append(prefix_message_prompt)

        # TODO: add support for example selectors
        for example in self._get_examples():
            message_prompts.append(
                ExampleHumanMessagePromptTemplate(
                    prompt=StringPromptTemplate.from_template(
                        self.example_input.format(**example)
                    )
                ),
            )
            message_prompts.append(
                ExampleAIMessagePromptTemplate(
                    prompt=StringPromptTemplate.from_template(
                        self.example_output.format(**example)
                    )
                ),
            )

        # construct the suffix message
        suffix_message_prompt = HumanMessagePromptTemplate(
            prompt=StringPromptTemplate.from_template(self.suffix)
        )
        message_prompts.append(suffix_message_prompt)

        chat_prompt_template = ChatPromptTemplate.from_messages(message_prompts)
        return chat_prompt_template.format(**kwargs)


if __name__ == "__main__":
    few_shot = FewShotChatPromptTemplate(
        prefix="You are a helpful assistant. You are helping translate from {source_language} to {target_language}.",
        suffix="{text}",
        example_input="{input}",
        example_output="{output}",
        examples=[
            {"input": "Hello", "output": "Bonjour"},
            {"input": "Goodbye", "output": "Au revoir"},
            {"input": "Thank you", "output": "Merci"},
            {"input": "I am sorry", "output": "Je suis désolé"},
        ],
        input_variables=["source_language", "target_language", "text"],
    )

    print(
        few_shot.format_prompt(
            source_language="English", target_language="French", text="How are you?"
        )
    )
