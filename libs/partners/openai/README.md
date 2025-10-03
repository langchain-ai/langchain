# langchain-openai

This package contains the LangChain integrations for OpenAI through their `openai` SDK.

## Installation and Setup

- Install the LangChain partner package

```bash
pip install langchain-openai
```

- Get an OpenAI api key and set it as an environment variable (`OPENAI_API_KEY`)

## Chat model

See a [usage example](https://python.langchain.com/docs/integrations/chat/openai).

```python
from langchain_openai import ChatOpenAI
```

### High concurrency - optional OpenAI aiohttp backend

For improved throughput in high-concurrency scenarios (parallel chains, graphs, and agents), you can enable the OpenAI aiohttp backend which removes concurrency limits seen with the default httpx client.

**Installation:**
```bash
pip install "openai[aiohttp]"
```

**Usage:**
```python
from openai import DefaultAioHttpClient
from langchain_openai import ChatOpenAI

# Option 1: Pass explicitly
llm = ChatOpenAI(
    http_client=DefaultAioHttpClient(),
    http_async_client=DefaultAioHttpClient()
)

# Option 2: Use environment variable
# Set LC_OPENAI_USE_AIOHTTP=1 in your environment
llm = ChatOpenAI()  # Will automatically use aiohttp if available
```

For more details, see the [OpenAI Python library documentation](https://github.com/openai/openai-python#httpx-client).

If you are using a model hosted on `Azure`, you should use different wrapper for that:

```python
from langchain_openai import AzureChatOpenAI
```

For a more detailed walkthrough of the `Azure` wrapper, see [AzureChatOpenAI](https://python.langchain.com/docs/integrations/chat/azure_chat_openai)

## Text Embedding Model

See a [usage example](https://python.langchain.com/docs/integrations/text_embedding/openai)

```python
from langchain_openai import OpenAIEmbeddings
```

If you are using a model hosted on `Azure`, you should use different wrapper for that:

```python
from langchain_openai import AzureOpenAIEmbeddings
```

For a more detailed walkthrough of the `Azure` wrapper, see [AzureOpenAIEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/azureopenai)

## LLM (Legacy)

LLM refers to the legacy text-completion models that preceded chat models. See a [usage example](https://python.langchain.com/docs/integrations/llms/openai).

```python
from langchain_openai import OpenAI
```

If you are using a model hosted on `Azure`, you should use different wrapper for that:

```python
from langchain_openai import AzureOpenAI
```

For a more detailed walkthrough of the `Azure` wrapper, see [Azure OpenAI](https://python.langchain.com/docs/integrations/llms/azure_openai)
