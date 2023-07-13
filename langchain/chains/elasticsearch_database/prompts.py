from langchain.output_parsers.list import CommaSeparatedListOutputParser
from langchain.prompts.prompt import PromptTemplate


PROMPT_SUFFIX = """Only use the following Elasticsearch indices:
{indices_info}

Question: {input}"""

# _DEFAULT_SQL_TEMPLATE = """Given an input question, first create a syntactically correct Elasticsearch SQL query to run, then look at the results of the query and return the answer. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

# Never query for all the columns from a specific index, only ask for a the few relevant columns given the question.

# Pay attention to use only the column names that you can see in the mapping description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which index.

# Use the following format:

# Question: Question here
# ESQuery: Elasticsearch SQL Query to run
# ESResult: Result of the Elasticsearch Query
# Answer: Final answer here

# """

# SQL_PROMPT = PromptTemplate(
#     input_variables=["input", "indices_info", "top_k"],
#     template=_DEFAULT_SQL_TEMPLATE + PROMPT_SUFFIX,
# )

_DEFAULT_DSL_TEMPLATE = """Given an input question, first create a syntactically correct Elasticsearch query to run, then look at the results of the query and return the answer. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific index, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the mapping description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which index.

Use the following format:

Question: Question here
ESQuery: Elasticsearch Query to run
ESResult: Result of the Elasticsearch Query
Answer: Final answer here

"""

DSL_PROMPT = PromptTemplate(
    input_variables=["input", "indices_info", "top_k"],
    template=_DEFAULT_DSL_TEMPLATE + PROMPT_SUFFIX,
)
