# langchain-nexa-ai

This package contains the LangChain integration with [Nexa AI](https://www.nexa4ai.com/).

## Installation

```bash
pip install -U langchain-nexa-ai
```

And you should get an api key from [Nexa Hub](https://hub.nexa4ai.com/api_keys) and set it as an environment variable (`NEXA_API_KEY`)

## LLMs

`NexaAILLM` class exposes Octopus LLMs from NexaAI. We currently support four catogories: `shopping`, `conference`, `streaming`, and `travel`. See our [langchain-nexa-ai tutorial](./docs/tutorial.ipynb) and [Nexa AI documentation](https://docs.nexa4ai.com/docs/overview) for more details!

```python
from langchain_nexa_ai import NexaAILLM

octopus_llm = NexaAILLM()
result = octopus_llm.invoke("Show recommended products for electronics.", category="shopping")
print(result)
```

If `NEXA_API_KEY` is not set in env, you can also pass api_key as an argument when initializing:

```python
octopus_llm = NexaAILLM(api_key=api_key)
```

You can also pass a list of catogories (corresponding to each of your prompts) when using `generate` method of `NexaAILLM`.

```python
result = octopus_llm.generate(
    prompts=[
        "Show recommended products for electronics.",
        "Find a Hotel to stay in Toyko from June 1 to June 10."
    ],
    categories=[
        "shopping",
        "travel"
    ]
)
```

### Exploit LCEL

```python
from langchain_nexa_ai import NexaAILLM
from langchain_core.output_parsers import JsonOutputParser

octopus_llm = NexaAILLM()
parser = JsonOutputParser()

chain = octopus_llm | parser

result = chain.invoke(
    input="Show me recommended electronics products",
    config={
        "llms": {
            "NexaAILLM": {
                "categories": ["shopping"]
            }
        }
    }
)
print(result)
```

## Chat Models

Coming soon.

## TODO

- [ ] streaming
- [ ] docs
