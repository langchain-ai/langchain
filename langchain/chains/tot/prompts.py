from textwrap import dedent

from langchain.prompts import PromptTemplate

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
)
