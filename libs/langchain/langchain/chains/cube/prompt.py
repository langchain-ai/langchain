# flake8: noqa
from langchain.output_parsers.list import CommaSeparatedListOutputParser
from langchain.prompts.prompt import PromptTemplate

PROMPT_SUFFIX = """Only use the following meta-information for Cubes defined in the data model:
{model_info}

Question: {input}"""

_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct Cube query to run, then look at the results of the query and return the answer. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific model, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

The current date is {today}.

Use the following format:

Question: Question here
CubeQuery: Cube Query to run
CubeResult: Result of the CubeQuery
Answer: Final answer here

{format_instructions}

"""

PROMPT = PromptTemplate(
    input_variables=["input", "model_info", "top_k","now"],
    template=_DEFAULT_TEMPLATE + PROMPT_SUFFIX,
)

_DECIDER_TEMPLATE = """Given the below input question and list of potential models, output a comma separated list of the model names that may be necessary to answer this question.

Question: {query}

Model Names: {model_names}

Relevant Model Names:"""
DECIDER_PROMPT = PromptTemplate(
    input_variables=["query", "model_names"],
    template=_DECIDER_TEMPLATE,
    output_parser=CommaSeparatedListOutputParser(),
)
