from langchain.schema import BaseMemory


def validate_sql_chain_memory(memory: BaseMemory) -> None:
    if memory.input_key is None:
        raise ValueError("You need to provide an input_key for SQLDatabaseChain Memory")
    if memory.input_key != "input":
        raise ValueError(f"Your memory_key should be input and not {memory.input_key}")
    if memory.memory_key != "history":
        raise ValueError(
            f"Your memory_key should be history and not {memory.memory_key}"
        )
