# langchain-xai

This package contains the LangChain integrations for [xAI](https://x.ai/) through their [APIs](https://console.x.ai).

## Installation and Setup

- Install the LangChain partner package

```bash
pip install -U langchain-xai
```

- Get your xAI api key from the [xAI Dashboard](https://console.x.ai) and set it as an environment variable (`XAI_API_KEY`)

## Chat Completions

This package contains the `ChatXAI` class, which is the recommended way to interface with xAI chat models.
