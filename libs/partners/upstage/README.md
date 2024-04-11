# langchain-upstage

This package contains the LangChain integrations for [Upstage](https://upstage.ai) through their [APIs](https://developers.upstage.ai/docs/getting-started/models).

## Installation and Setup

- Install the LangChain partner package
```bash
pip install -U langchain-upstage
```

- Get an Upstage api key from [Upstage Console](https://console.upstage.ai/home) and set it as an environment variable (`UPSTAGE_API_KEY`)

## Chat Models

This package contains the `ChatUpstage` class, which is the recommended way to interface with Upstage models.

See a [usage example](https://python.langchain.com/docs/integrations/chat/upstage)

## Embeddings

See a [usage example](https://python.langchain.com/docs/integrations/text_embedding/upstage)

Use `solar-1-mini-embedding` as the default model for embeddings. Do not add suffixes such as `-query` or `-passage` to the model name.
`UpstageEmbeddings` will automatically add the suffixes based on the method called.
