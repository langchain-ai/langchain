# Fireworks

This page covers how to use [Fireworks](https://app.fireworks.ai/) models within
Langchain.

## Installation and setup

- Install the Fireworks client library.

  ```
  pip install fireworks-ai
  ```

- Get a Fireworks API key by signing up at [app.fireworks.ai](https://app.fireworks.ai).
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
will work the llama-v2-13b-chat model. 

```python
from langchain_community.llms.fireworks import Fireworks 

llm = Fireworks(
    fireworks_api_key="<KEY>",
    model="accounts/fireworks/models/llama-v2-13b-chat",
    max_tokens=256)
llm("Name 3 sports.")
```

For a more detailed walkthrough, see [here](/docs/integrations/llms/Fireworks).
