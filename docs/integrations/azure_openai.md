# Azure OpenAI

>[Microsoft Azure](https://en.wikipedia.org/wiki/Microsoft_Azure), often referred to as `Azure` is a cloud computing platform run by `Microsoft`, which offers access, management, and development of applications and services through global data centers. It provides a range of capabilities, including software as a service (SaaS), platform as a service (PaaS), and infrastructure as a service (IaaS). `Microsoft Azure` supports many programming languages, tools, and frameworks, including Microsoft-specific and third-party software and systems.


>[Azure OpenAI](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/) is an `Azure` service with powerful language models from `OpenAI` including the `GPT-3`, `Codex` and `Embeddings model` series for content generation, summarization, semantic search, and natural language to code translation.


## Installation and Setup

```bash
pip install openai
pip install tiktoken
```


Set the environment variables to get access to the `Azure OpenAI` service.

```python
import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://<your-endpoint.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "your AzureOpenAI key"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
```

## LLM

See a [usage example](../modules/models/llms/integrations/azure_openai_example.ipynb).

```python
from langchain.llms import AzureOpenAI
```

## Text Embedding Models

See a [usage example](../modules/models/text_embedding/examples/azureopenai.ipynb)

```python
from langchain.embeddings import OpenAIEmbeddings
```

## Chat Models

See a [usage example](../modules/models/chat/integrations/azure_chat_openai.ipynb)

```python
from langchain.chat_models import AzureChatOpenAI
```
