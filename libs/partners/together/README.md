# langchain-together

This package contains the LangChain integrations for [Together AI](https://www.together.ai/) through their [APIs](https://docs.together.ai/).

## Installation and Setup

- Install the LangChain partner package

```bash
pip install -U langchain-together
```

- Get your Together AI api key from the [Together Dashboard](https://api.together.ai/settings/api-keys) and set it as an environment variable (`TOGETHER_API_KEY`)

## Chat Completions

This package contains the `ChatTogether` class, which is the recommended way to interface with Together AI chat models.

ADD USAGE EXAMPLE HERE.
Can we add this in the langchain docs?

NEED to add image endpoint + completions endpoint as well

## Embeddings

See a [usage example](https://python.langchain.com/docs/integrations/text_embedding/together/)

Use `togethercomputer/m2-bert-80M-8k-retrieval` as the default model for embeddings.
