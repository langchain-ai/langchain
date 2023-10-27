# Get LLM
import os
from pathlib import Path

import requests
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.utilities import SQLDatabase

# File name and URL
file_name = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
# Check if file is present in the current directory
if not os.path.exists(file_name):
    print(f"'{file_name}' not found. Downloading...")
    # Download the file
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    with open(file_name, "wb") as f:
        f.write(response.content)
    print(f"'{file_name}' has been downloaded.")
else:
    print(f"'{file_name}' already exists in the current directory.")

# Add the LLM downloaded from HF
model_path = file_name
n_gpu_layers = 1  # Metal set to 1 is enough.

# Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
n_batch = 512

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    # f16_kv MUST set to True
    # otherwise you will run into problem after a couple of calls
    f16_kv=True,
    verbose=True,
)

db_path = Path(__file__).parent / "nba_roster.db"
rel = db_path.relative_to(Path.cwd())
db_string = f"sqlite:///{rel}"
db = SQLDatabase.from_uri(db_string, sample_rows_in_table_info=0)


def get_schema(_):
    return db.get_table_info()


def run_query(query):
    return db.run(query)


# Prompt

template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""  # noqa: E501
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Given an input question, convert it to a SQL query. No pre-amble."),
        MessagesPlaceholder(variable_name="history"),
        ("human", template),
    ]
)

memory = ConversationBufferMemory(return_messages=True)

# Chain to query with memory

sql_chain = (
    RunnablePassthrough.assign(
        schema=get_schema,
        history=RunnableLambda(lambda x: memory.load_memory_variables(x)["history"]),
    )
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)


def save(input_output):
    output = {"output": input_output.pop("output")}
    memory.save_context(input_output, output)
    return output["output"]


sql_response_memory = RunnablePassthrough.assign(output=sql_chain) | save

# Chain to answer
template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""  # noqa: E501
prompt_response = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question and SQL response, convert it to a natural "
            "language answer. No pre-amble.",
        ),
        ("human", template),
    ]
)

chain = (
    RunnablePassthrough.assign(query=sql_response_memory)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | llm
)
