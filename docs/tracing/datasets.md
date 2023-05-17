# Datasets

This guide provides instructions for how to use Datasets generated from LangChain traces.

Some things datasets are useful for include:

- Compare the results of different models or prompts to pick the most appropriate configuration.
- Test for regressions in LLM or agent behavior over known use cases.
- Run a model N times to measure the stability of its predictions and infer the reliability of its performance.
- Run an evaluation chain over your agents' outputs to quantify your agents' performance.


## Creating a Dataset

Datasets store the inputs and outputs of LLM, chat model, or chain or agent runs.

### Using the LangChainPlusClient

To create the client:

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_SESSION"] = "Tracing Walkthrough"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"  # Uncomment this line if you want to use the hosted version
# os.environ["LANGCHAIN_API_KEY"] = "<YOUR-LANGCHAINPLUS-API-KEY>"  # Uncomment this line if you want to use the hosted version.

client = LangChainPlusClient()
```

There are two main ways to create datasets:

    1. From a CSV or pandas DataFrame

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
   
   2. From traced runs. Assuming you've already captured runs in a session called "My Agent Session":

    ```python
    runs = client.list_runs(session_name="My Agent Session")
    dataset = client.create_dataset("My Dataset", "Examples from My Agent")
    for run in runs:
        client.create_example(inputs=run.inputs, outputs=run.outputs, dataset_id=dataset.id)
    ```


### Using the UI


## Running LangChain objects on datasets

