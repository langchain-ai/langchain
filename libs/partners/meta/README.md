# langchain-meta

This package contains the LangChain integrations for [Meta](https://llama.com/) through their [APIs](https://llama.developer.meta.com?utm_source=partner-langchain&utm_medium=readme).

## Installation and Setup

- Install the LangChain partner package

```bash
pip install -U langchain-meta
```

- Get your Llama api key from the [Meta](https://llama.developer.meta.com?utm_source=partner-langchain&utm_medium=readme) and set it as an environment variable (`LLAMA_API_KEY`)

## Chat Completions

This package contains the `ChatLlama` class, which is the recommended way to interface with Llama chat models.

## API Models

| Model ID | Input context length | Output context length | Input Modalities | Output Modalities |
| --- | --- | --- | --- | --- |
| `Llama-4-Scout-17B-16E-Instruct-FP8` | 128k | 4028 | Text, Image | Text |
| `Llama-4-Maverick-17B-128E-Instruct-FP8` | 128k | 4028 | Text, Image | Text |
| `Llama-3.3-70B-Instruct` | 128k | 4028 | Text | Text |
| `Llama-3.3-8B-Instruct` | 128k | 4028 | Text | Text |
