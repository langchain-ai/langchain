# Fireworks

This page covers how to use the Fireworks models within Langchain.

## Installation and Setup

- To use the Fireworks model, you need to have a Fireworks API key. To generate one, sign up at [app.fireworks.ai](https://app.fireworks.ai).
- Authenticate by setting the FIREWORKS_API_KEY environment variable.

## LLM

Fireworks integrates with Langchain through the LLM module, which allows for standardized usage of any models deployed on the Fireworks models.

In this example, we'll work the llama-v2-13b-chat model. 

```python
from langchain.llms.fireworks import Fireworks 

llm = Fireworks(model="fireworks-llama-v2-13b-chat", max_tokens=256, temperature=0.4)
llm("Name 3 sports.")
```

For a more detailed walkthrough, see [here](/docs/integrations/llms/Fireworks).
