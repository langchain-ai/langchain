from langchain.output_parsers import GuardrailsOutputParser
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate

# Define rail string

rail_str = """
<rail version="0.1">
<output>
    <string 
        description="Profanity-free translation" 
        format="is-profanity-free" 
        name="translated_statement" 
        on-fail-is-profanity-free="fix">
    </string>
</output>
<prompt>
    Translate the given statement into English:

    ${statement_to_be_translated}

    ${gr.complete_json_suffix}
</prompt>
</rail>
"""


# Create the GuardrailsOutputParser object from the rail string
output_parser = GuardrailsOutputParser.from_rail_string(rail_str)

# Define the prompt, model and chain
prompt = PromptTemplate(
    template=output_parser.guard.prompt.escape(),
    input_variables=output_parser.guard.prompt.variable_names,
)

chain = prompt | OpenAI() | output_parser

# This is needed because GuardrailsOutputParser does not have an inferrable type
chain = chain.with_types(output_type=dict)
