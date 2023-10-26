from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.output_parsers import GuardrailsOutputParser

from .rails import RAIL_STRING

output_parser = GuardrailsOutputParser.from_rail_string(RAIL_STRING)
prompt = PromptTemplate(
    template=output_parser.guard.prompt.escape(),
    input_variables=output_parser.guard.prompt.variable_names,
)
model = OpenAI(temperature=0)

chain = prompt | model | output_parser
