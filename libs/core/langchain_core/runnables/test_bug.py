from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig

from langchain_core.runnables import configurable


def inspect_config(x , config : RunnableConfig):
    print("Config Received  :")
    print(f"configurable : {config.get('configurable' , {})} ")
    print(f"metadata : {config.get('metadata', {})}")

    if 'user_id' in config.get('configurable' ,{}):
        if 'user_id' not in config.get('metadata', {}):
            print("  BUG: user_id is in configurable but NOT in metadata!")
    return x

runnable = RunnableLambda(inspect_config)

config = {
    "configurable": {"user_id": "alice", "session_id": "123"},
    "metadata": {"source": "test"}
}


print("Testing with invoke:")
result = runnable.invoke("test input", config=config) # pyright: ignore[reportArgumentType]

print("\nTesting with stream:")
for chunk in runnable.stream("test input", config=config): # pyright: ignore[reportArgumentType]
    pass


