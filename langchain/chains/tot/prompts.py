import json
import re
from textwrap import dedent

from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser


class NextStepOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        """Parse the output of the language model."""
        if match := re.search(r"\{.*?\}", text, re.DOTALL):
            try:
                return json.loads(match.group())["next_step"]
            except json.JSONDecodeError:
                return ""
        return ""

    @property
    def _type(self) -> str:
        return "next_step_output"


FIRST_STEP_PROMPT = PromptTemplate(
    input_variables=["problem_description"],
    template=dedent(
        """
        For the given problem:
        
        {problem_description}
        
        Please derive the first step, and return the step in the following JSON
        format {{"next_step": "<next_step>"}}
        """
    ),
    output_parser=NextStepOutputParser(),
)


NEXT_STEP_PROMPT = PromptTemplate(
    input_variables=["problem_description", "partial_solution_summary"],
    template=dedent(
        """
        For the given problem: 
        
        {problem_description}
        
        We have come up with the a partial solution: 
        
        {partial_solution_summary}
        
        Please derive the next step on top of this partial solution, and return
        the step in the following JSON format {{"next_step": "<next_step>"}}
        """
    ),
    output_parser=NextStepOutputParser(),
)
