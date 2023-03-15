from __future__ import annotations

from typing import Dict

try:
    from guardrails import Guard
except ImportError:
    pass


from langchain.output_parsers.base import BaseOutputParser
from langchain.prompts import PromptTemplate


class GuardrailOutputParser(BaseOutputParser):
    guard = Guard

    @property
    def _type(self) -> str:
        return "guardrail"

    @classmethod
    def from_rail(cls, rail_file: str, num_reasks: int = 1) -> GuardrailOutputParser:
        return cls(guard=Guard.from_rail(rail_file, num_reasks=num_reasks))

    @classmethod
    def from_rail_string(cls, rail_str: str, num_reasks: int = 1) -> GuardrailOutputParser:
        return cls(guard=Guard.from_rail_string(rail_str, num_reasks=num_reasks))

    def get_format_instructions(self) -> str:
        return self.guard.raw_prompt.format_instructions

    def to_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template=self.guard.base_prompt,
            input_variables=self.guard.prompt.input_variables
        )

    def parse(self, text: str) -> Dict:
        return self.guard.parse(text)
