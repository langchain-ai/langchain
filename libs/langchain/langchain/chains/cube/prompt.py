# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

PROMPT_SUFFIX = """Only use the following meta-information for Cubes defined in the data model:
{model_meta_information}
Question: {input}"""

_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct Cube query to run, then look at the results of the query and return the answer. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most relevant examples in the database.
Never query for all the columns from a specific model, only ask for a the few relevant columns given the question.
Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
The current date is {current_date}.

Use the following format:
Question: Question here
CubeQuery: Cube Query to run
CubeResult: Result of the CubeQuery
Answer: Final answer here
{format_instructions}"""

PROMPT = PromptTemplate(
    input_variables=["input", "model_meta_information", "top_k", "current_date"],
    template=_DEFAULT_TEMPLATE + PROMPT_SUFFIX,
)
