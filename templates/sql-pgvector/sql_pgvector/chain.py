import os
import re

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.sql_database import SQLDatabase

from prompt_templates import postgresql_template, final_template


"""
# TODO explain how to setup
"""

if os.environ.get("OPENAI_API_KEY", None) is None:
    raise Exception("Missing `OPENAI_API_KEY` environment variable")


# Connect to DB
CONNECTION_STRING = "postgresql+psycopg2://postgres:test@localhost:5432/vectordb"  # Replace with your own
db = SQLDatabase.from_uri(CONNECTION_STRING)

# Choose LLM and embeddings model
llm = ChatOpenAI(temperature=0)
embeddings_model = OpenAIEmbeddings()

#-----------------
# Define functions
#-----------------
def get_schema(_):
    return db.get_table_info()

def run_query(query):
    return db.run(query)

def replace_brackets(match):
    words_inside_brackets = match.group(1).split(", ")
    embedded_words = [
        str(embeddings_model.embed_query(word)) for word in words_inside_brackets
    ]
    return "', '".join(embedded_words)


def get_query(query):
    sql_query = re.sub(r"\[([\w\s,]+)\]", replace_brackets, query)
    return sql_query

# -----------------------
# Now we create the chain
# -----------------------

query_genertion_prompt = ChatPromptTemplate.from_messages(
    [("system", postgresql_template), ("human", "{question}")]
)

sql_query_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | query_genertion_prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)


final_prompt = ChatPromptTemplate.from_messages(
    [("system", final_template), ("human", "{question}")]
)

full_chain = (
    RunnablePassthrough.assign(query=sql_query_chain)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=RunnableLambda(lambda x: db.run(get_query(x["query"]))),
    )
    | final_prompt
    | llm
)

chain = full_chain