# Pipeshift

> [Pipeshift](https://pipeshift.com) is a fine-tuning and inference platform for open-source LLMs

- You bring your datasets. Fine-tune multiple LLMs. Start inferencing in one-click and watch them scale to millions.

## Installation and Setup

- Install the Pipeshift integration package.

  ```
  pip install langchain-pipeshift
  ```

- Get your Pipeshift API key by signing up at [Pipeshift](https://pipeshift.com).

### Authentication

You can perform authentication using your Pipeshift API key in any of the following ways:

1.  Adding API key to the environment variable as `PIPESHIFT_API_KEY`.

    ```python
    os.environ["PIPESHIFT_API_KEY"] = "<your_api_key>"
    ```

2.  By passing `api_key` to the pipeshift LLM module or chat module

    ```python
    llm = Pipeshift(api_key="<your_api_key>", model="meta-llama/Meta-Llama-3.1-8B-Instruct", max_tokens=512)

                        OR

    chat = ChatPipeshift(api_key="<your_api_key>", model="meta-llama/Meta-Llama-3.1-8B-Instruct", max_tokens=512)
    ```

## Chat models

See an [example](/docs/integrations/chat/pipeshift).

```python
from langchain_pipeshift import ChatPipeshift
```

## LLMs

See an [example](/docs/integrations/llms/pipeshift).

```python
from langchain_pipeshift import Pipeshift
```
