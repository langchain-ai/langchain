# Fireworks

This page covers how to use [Fireworks](https://fireworks.ai/) models within
Langchain.

## Installation and setup

- Install the Fireworks integration package.

  ```
  pip install langchain-fireworks
  ```

- Get a Fireworks API key by signing up at [fireworks.ai](https://fireworks.ai).
- Authenticate by setting the FIREWORKS_API_KEY environment variable.

## Authentication

There are two ways to authenticate using your Fireworks API key:

1.  Setting the `FIREWORKS_API_KEY` environment variable.

    ```python
    os.environ["FIREWORKS_API_KEY"] = "<KEY>"
    ```

2.  Setting `fireworks_api_key` field in the Fireworks LLM module.

    ```python
    llm = Fireworks(fireworks_api_key="<KEY>")
    ```

## Using the Fireworks LLM module

Fireworks integrates with Langchain through the LLM module. In this example, we
will work the mixtral-8x7b-instruct model. 

```python
from langchain_fireworks import Fireworks 

llm = Fireworks(
    fireworks_api_key="<KEY>",
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
    max_tokens=256)
llm("Name 3 sports.")
```

For a more detailed walkthrough, see [here](/docs/integrations/llms/Fireworks).
