from langchain.prompts.base import BasePrompt, DEFAULT_FORMATTER_MAPPING
from pydantic import BaseModel, Extra, root_validator
from langchain.prompts.prompt import Prompt
from typing import List, Any

class FewShotPrompt(BasePrompt, BaseModel):

    examples: List[dict]
    example_prompt: Prompt
    suffix: str
    input_variables: List[str]
    example_separator: str = "\n\n"
    prefix: str = ""
    template_format: str = "f-string"


    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _get_examples(self, **kwargs: Any):
        # TODO: add ExampleSelector logic here
        return self.examples

    def format(self, **kwargs: Any) -> str:
        examples = self._get_examples(**kwargs)
        example_strings = [self.example_prompt.format(**example) for example in examples]
        pieces = [self.prefix, *example_strings, self.suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)


