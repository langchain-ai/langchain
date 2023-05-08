from __future__ import annotations

import attr
from textwrap import dedent
from langchain.concise.config import get_default_text_splitter
from pydantic import BaseModel, Field, validator

from langchain.base_language import BaseLanguageModel
from langchain.concise.generate import generate
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.choice import ChoiceOutputParser
from langchain.output_parsers.item_parsed_list import ItemParsedListOutputParser
from langchain.schema import OutputParserException
from langchain.text_splitter import TextSplitter, TokenTextSplitter


class Rule(BaseModel):
    name: str = Field(description="Name of the rule.")
    pattern: str = Field(description="Natural language description of what to replace.")
    replacement: str = Field(
        description="Natural language description of what to replace it with or how to rewrite it."
    )

    def __str__(self) -> str:
        return f"{self.name}: {self.pattern} -> {self.replacement}"


class RulEx(BaseModel):
    _NO_RULE_MATCH = "No match"

    rules: list[Rule]
    choice_parser: ChoiceOutputParser
    text_splitter: TextSplitter
    replacements_per_chunk: int = 5

    @classmethod
    def create(
        cls,
        rules: str | list[str] | list[tuple[str, str]] | list[Rule],
        text_splitter: TextSplitter,
        llm: BaseLanguageModel,
    ) -> RulEx:
        rules = cls._parse_rules(rules, llm)
        text_splitter = text_splitter or get_default_text_splitter()
        choice_parser = ChoiceOutputParser(
            options=[rule.pattern for rule in rules] + [RulEx._NO_RULE_MATCH], llm=llm
        )
        return cls(rules=rules, text_splitter=text_splitter, choice_parser=choice_parser)

    @classmethod
    def _parse_rules(cls, rules, llm):
        rule_parser = PydanticOutputParser(pydantic_object=Rule)
        if isinstance(rules, list) and all(isinstance(rule, Rule) for rule in rules):
            pass
        elif (
            isinstance(rules, list)
            and all(isinstance(rule, tuple) for rule in rules)
            and all(len(rule) == 2 for rule in rules)
        ):
            rules = [
                Rule(name=f"Rule_{i}", pattern=pattern, replacement=replacement)
                for i, (pattern, replacement) in enumerate(rules)
            ]
        elif isinstance(rules, list) and all(isinstance(rule, str) for rule in rules):
            rules = [rule_parser.parse(rule) for rule in rules]
        elif isinstance(rules, str):
            rules = generate(
                "Please extract a list of natural language replacement rules from the following text: {% for rule in rules %}- (rules)\n{% endfor %}",
                llm=llm,
                parser=ItemParsedListOutputParser(
                    item_parser=rule_parser, item_name="rule"
                ),
            )
        else:
            raise ValueError(
                f"Invalid rule type: {type(rules)}. Should be a list of strings, a list of tuples, or a list of Rule objects."
            )
        return rules

    def __call__(self, input) -> str:
        output = ""
        for chunk in self.text_splitter.chunk(input):
            for _ in range(self.replacements_per_chunk):
                # get the next matching rule
                rule_name = generate(
                    f"Select the first rule that matches the following text:\n\n{chunk}\n\n",
                    llm=self.llm,
                    parser=self.choice_parser,
                )
                if rule_name == self._NO_RULE_MATCH:
                    break
                rule = next(rule for rule in self.rules if rule.name == rule_name)

                # identify the exact match to replace
                def no_match_found(output):
                    if output not in chunk:
                        raise OutputParserException("No exact match found.")

                match = generate(
                    "Identify the string exact match to replace.",
                    llm=self.llm,
                    additional_validator=no_match_found,
                )
                chunk_with_match_selection = chunk.replace(match, f"[[{match}]]")
                replacement = generate(
                    dedent(
                        f"""
                        Instructions: Write the new text that should replace \"[[{match}]]\" from the input below.
                        
                        Replace {rule.pattern} with {rule.replacement}.
                        
                        Input: "{chunk_with_match_selection}"
                        
                        Replacement value: 
                        """
                    ),
                    llm=self.llm,
                ).strip()
                chunk = chunk.replace(match, replacement)
            output += chunk
        return output
