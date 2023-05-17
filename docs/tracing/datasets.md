# Datasets

This guide provides instructions on how to use traced Datasets in LangChain Plus.

Datasets are broadly useful for developing and productionizing LLM applications, as they enable:

- comparing the results of different models or prompts to pick the most appropriate configuration.
- tesing for regressions in LLM or agent behavior over known use cases.
- running a model N times to measure the stability of its predictions and infer the reliability of its performance.
- running an evaluation chain over your agents' outputs to quantify your agents' performance.

As well as many other applications.


## Creating a Dataset

Datasets store examples holding the inputs and outputs (or 'ground truth' labels) of LLM, chat model, or chain or agent runs.
You can create datasets using the `LangChainPlusClient` (which connects to the tracing server's REST API) or in the UI.

## Using the UI

You can directly create Datasets in the UI in two ways:

    - **Upload data from a CSV file.**
        1. Click on the `Datasets` page in the LangChain Plus homepage or click `Menu` in the top right-hand corner and click 'Datasets'
        2. Click "Upload CSV"
        3. Upload the CSV, and specify the column names that represent the LLM or Chain's inputs and outputs
        <!-- TODO: Add a screenshot -->
    - **Convert traced runs to a Dataset**
        1. Navigate to a Session containing runs.
        2. For rows you wish to add, click on the "+" sign on the right-hand side of the row to "Add Example to Dataset"
        3. Either "Create dataset" or select an existing one to design where to add. If you wish to update the expected output, you can updates the text in the box.
        <!-- TODO: Add screenshots -->

## Using the LangChainPlusClient 


The `LangChainPlusClient` connects to the tracing server's REST API. For more information on the client, please reference the [LangChain Plus Client](./langchain_plus_client.md) guide.

To create a client:

```python

# import os
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"  # Uncomment this line if you want to use the hosted version
# os.environ["LANGCHAIN_API_KEY"] = "<YOUR-LANGCHAINPLUS-API-KEY>"  # Uncomment this line if you want to use the hosted version.

from langchain.client import LangChainPlusClient

client = LangChainPlusClient()
```

### Datasets and the LangChainPlusClient

The following are two simple ways to create a dataset with the client:

    -  **Upload data from a CSV or pandas DataFrame**

    ```python
    csv_path = "path/too/data.csv"
    input_keys = ["input"] # column names that will be input to Chain or LLM
    output_keys = ["output"] # column names that are the output of the Chain or LLM
    description = "My dataset for evaluation"
    dataset = client.upload_csv(
        csv_path,
        description=description,
        input_keys=input_keys,
        output_keys=output_keys,
    )
    # Or as a DataFrame
    import pandas as pd
    df = pd.read_csv(csv_path)
    dataset = client.upload_dataframe(
        df, 
        "My Dataset", 
        description=description,
        input_keys=input_keys,
        output_keys=output_keys,
    )
    ```
   
   -  **Create a dataset from traced runs.** Assuming you've already captured runs in a session called "My Agent Session":

    ```python
    runs = client.list_runs(session_name="My Agent Session", error=False) # List runs in my session that don't have errors
    dataset = client.create_dataset("My Dataset", "Examples from My Agent")
    for run in runs:
        client.create_example(inputs=run.inputs, outputs=run.outputs, dataset_id=dataset.id)
    ```


## Using Datasets

The `LangChainPlusClient` can help you flexibly run any LangChain object over your datasets. Below are a few common use cases.
Before running any of these examples, make sure you have created your client and datasets using one of the methods above.

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_SESSION"] = "Tracing Walkthrough"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"  # Uncomment this line if you want to use the hosted version
# os.environ["LANGCHAIN_API_KEY"] = "<YOUR-LANGCHAINPLUS-API-KEY>"  # Uncomment this line if you want to use the hosted version.

client = LangChainPlusClient()
```

### Running LLMs over Datasets

Once you've created a dataset (we'll call it "LLM Dataset" here) with a string prompt input and generated outputs, you can
compare results by specifying other LLMs and running over the saved dataset.

```python
from langchain import OpenAI
dataset_name = "LLM Dataset" # Update to the correct dataset

llm = OpenAI(temperature=0)
llm_results = await client.arun_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=llm,
)
```

The traces from this run will be saved in a new session linked to the dataset, and the model outputs
will be returned. You can also run the LLM synchronously if async isn't supported
(though this will likely take longer).

```python
Or to run the LLM synchronously
llm_results = client.run_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=llm,
)
```

You can then view the UI to see the run results in a new session.

### Running Chat Models over Datasets

You can run Chat Models over datasets captured from LLM or Chat Model runs as well.

```python
from langchain.chat_models import ChatOpenAI

dataset_name = "Chat Model Dataset"
llm = OpenAI(temperature=0)
llm_results = await client.arun_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=llm,
)
```

The synchronous `client.run_on_dataset` method is also available for chat models.

### Running Chains over Datasets


You can also run any chain or agent over stored datasets to do things like evaluate outputs and compare prompts, models, and tool usage.

Many chains contain `memory`, so to treat each example independently, we have to pass in a "chain factory" (or constructor) that tells the
client how to create the chain. This also means that chains that interact with remote/persistant storage must be configured appropriately to
avoid using the same memory across each example. If you know your chain does _not_ use memory, this factory can be a simple lambda (`lambda: my_agent`) that avoids
re-creating objects


```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, load_tools
from langchain.agents import AgentType

llm = ChatOpenAI(temperature=0)
tools = load_tools(['serpapi', 'llm-math'], llm=llm)
agent_factory = lambda : initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

dataset_name = "Agent Dataset"

agent_results = await client.arun_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=agent_factory,
)
```