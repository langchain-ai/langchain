# flake8: noqa
from langchain_core.prompts.prompt import PromptTemplate

PROMPT_SUFFIX = """Only use the following Elasticsearch indices:
{indices_info}

Question: {input}
ESQuery:"""

DEFAULT_DSL_TEMPLATE = """Given an input question, create a syntactically correct Elasticsearch query to run. Unless the user specifies in their question a specific number of examples they wish to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

Unless told to do not query for all the columns from a specific index, only ask for a few relevant columns given the question.

Pay attention to use only the column names that you can see in the mapping description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which index. Return the query as valid json.

Use the following format:

Question: Question here
ESQuery: Elasticsearch Query formatted as json
"""

DSL_PROMPT = PromptTemplate.from_template(DEFAULT_DSL_TEMPLATE + PROMPT_SUFFIX)

DEFAULT_ANSWER_TEMPLATE = """Given an input question and relevant data from a database, answer the user question.

Use the following format:

Question: Question here
Data: Relevant data here
Answer: Final answer here

Question: {input}
Data: {data}
Answer:"""

ANSWER_PROMPT = PromptTemplate.from_template(DEFAULT_ANSWER_TEMPLATE)
