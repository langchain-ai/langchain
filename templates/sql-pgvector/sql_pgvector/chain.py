import os
import re

from langchain.sql_database import SQLDatabase
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from sql_pgvector.prompt_templates import final_template, postgresql_template

"""
IMPORTANT: For using this template, you will need to 
follow the setup steps in the readme file
"""

if os.environ.get("OPENAI_API_KEY", None) is None:
    raise Exception("Missing `OPENAI_API_KEY` environment variable")

postgres_user = os.environ.get("POSTGRES_USER", "postgres")
postgres_password = os.environ.get("POSTGRES_PASSWORD", "test")
postgres_db = os.environ.get("POSTGRES_DB", "vectordb")
postgres_host = os.environ.get("POSTGRES_HOST", "localhost")
postgres_port = os.environ.get("POSTGRES_PORT", "5432")

# Connect to DB
# Replace with your own
CONNECTION_STRING = (
    f"postgresql+psycopg2://{postgres_user}:{postgres_password}"
    f"@{postgres_host}:{postgres_port}/{postgres_db}"
)
db = SQLDatabase.from_uri(CONNECTION_STRING)

# Choose LLM and embeddings model
llm = ChatOpenAI(temperature=0)
embeddings_model = OpenAIEmbeddings()


# # Ingest code - you will need to run this the first time
# # Insert your query e.g. "SELECT Name FROM Track"
# column_to_embed = db.run('replace-with-your-own-select-query')
# column_values = [s[0] for s in eval(column_to_embed)]
# embeddings = embeddings_model.embed_documents(column_values)

# for i in range(len(embeddings)):
#     value = column_values[i].replace("'", "''")
#     embedding = embeddings[i]

#     # Replace with your own SQL command for your column and table.
#     sql_command = (
#         f'UPDATE "Track" SET "embeddings" = ARRAY{embedding} WHERE "Name" ='
#         + f"'{value}'"
#     )
#     db.run(sql_command)


# -----------------
# Define functions
# -----------------
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

query_generation_prompt = ChatPromptTemplate.from_messages(
    [("system", postgresql_template), ("human", "{question}")]
)

sql_query_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | query_generation_prompt
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


class InputType(BaseModel):
    question: str


chain = full_chain.with_types(input_type=InputType)
