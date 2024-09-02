# LiteLLM

>[LiteLLM](https://docs.litellm.ai/docs/) is a library that simplifies calling Anthropic, 
> Azure, Huggingface, Replicate, etc. LLMs in a unified way.
> 
>You can use `LiteLLM` through either:
>
>* [LiteLLM Proxy Server](https://docs.litellm.ai/docs/#openai-proxy) - Server to call 100+ LLMs, load balance, cost tracking across projects
>* [LiteLLM python SDK](https://docs.litellm.ai/docs/#basic-usage) - Python Client to call 100+ LLMs, load balance, cost tracking

## Installation and setup

Install the `litellm` python package.

```bash
pip install litellm
```

## Chat models

### ChatLiteLLM

See a [usage example](/docs/integrations/chat/litellm).

```python
from langchain_community.chat_models import ChatLiteLLM
```

### ChatLiteLLMRouter

You also can use the `ChatLiteLLMRouter` to route requests to different LLMs or LLM providers.

See a [usage example](/docs/integrations/chat/litellm_router).

```python
from langchain_community.chat_models import ChatLiteLLMRouter
```
