"""utils module consist of function which validates memory passed to sql agent/chain"""

from langchain.schema import BaseMemory, BasePromptTemplate


def validate_sql_chain_memory(memory: BaseMemory, prompt: BasePromptTemplate) -> None:
    if not memory.input_key:
        raise ValueError(
            "You need to provide an input_key for memory to work with sql-database chain/agent"
        )
    if memory.memory_key not in prompt.input_variables:
        raise ValueError(
            f"You need to provide {memory.memory_key} as input variable in your PromptTemplate"
        )
