from langchain.schema import BaseMemory


def validate_sql_chain_memory(memory:BaseMemory) -> None:
    if memory.input_key is None:
        raise ValueError("You need to provide an input_key for SQLDatabaseChain Memory")