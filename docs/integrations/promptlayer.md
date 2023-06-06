# PromptLayer

>[PromptLayer](https://docs.promptlayer.com/what-is-promptlayer/wxpF9EZkUwvdkwvVE9XEvC/how-promptlayer-works/dvgGSxNe6nB1jj8mUVbG8r) 
> is a devtool that allows you to track, manage, and share your GPT prompt engineering. 
> It acts as a middleware between your code and OpenAI's python library, recording all your API requests 
> and saving relevant metadata for easy exploration and search in the [PromptLayer](https://www.promptlayer.com) dashboard.

## Installation and Setup

- Install the `promptlayer` python library 
```bash
pip install promptlayer
```
- Create a PromptLayer account
- Create an api token and set it as an environment variable (`PROMPTLAYER_API_KEY`)


## LLM

```python
from langchain.llms import PromptLayerOpenAI
```

### Example

To tag your requests, use the argument `pl_tags` when instantiating the LLM
```python
from langchain.llms import PromptLayerOpenAI
llm = PromptLayerOpenAI(pl_tags=["langchain-requests", "chatbot"])
```

To get the PromptLayer request id, use the argument `return_pl_id` when instantiating the LLM
```python
from langchain.llms import PromptLayerOpenAI
llm = PromptLayerOpenAI(return_pl_id=True)
```
This will add the PromptLayer request ID in the `generation_info` field of the `Generation` returned when using `.generate` or `.agenerate`

For example:
```python
llm_results = llm.generate(["hello world"])
for res in llm_results.generations:
    print("pl request id: ", res[0].generation_info["pl_request_id"])
```
You can use the PromptLayer request ID to add a prompt, score, or other metadata to your request. [Read more about it here](https://magniv.notion.site/Track-4deee1b1f7a34c1680d085f82567dab9).

This LLM is identical to the [OpenAI LLM](./openai.md), except that
- all your requests will be logged to your PromptLayer account
- you can add `pl_tags` when instantiating to tag your requests on PromptLayer
- you can add `return_pl_id` when instantiating to return a PromptLayer request id to use [while tracking requests](https://magniv.notion.site/Track-4deee1b1f7a34c1680d085f82567dab9).

## Chat Model

```python
from langchain.chat_models import PromptLayerChatOpenAI
```

See a [usage example](../modules/models/chat/integrations/promptlayer_chatopenai.ipynb).

