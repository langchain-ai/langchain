# langchain-maritaca

[![PyPI version](https://badge.fury.io/py/langchain-maritaca.svg)](https://badge.fury.io/py/langchain-maritaca)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An integration package connecting [Maritaca AI](https://www.maritaca.ai/) and [LangChain](https://langchain.com/) for Brazilian Portuguese language models.

**Author:** Anderson Henrique da Silva
**Location:** Minas Gerais, Brasil
**GitHub:** [anderson-ufrj](https://github.com/anderson-ufrj)

## Overview

Maritaca AI provides state-of-the-art Brazilian Portuguese language models, including the Sabiá family of models. This integration allows you to use Maritaca's models seamlessly within the LangChain ecosystem.

### Available Models

| Model | Description | Pricing (per 1M tokens) |
|-------|-------------|------------------------|
| `sabia-3` | Most capable model, best for complex tasks | R$ 5.00 input / R$ 10.00 output |
| `sabiazinho-3` | Fast and economical, great for simple tasks | R$ 1.00 input / R$ 3.00 output |

## Installation

```bash
pip install -U langchain-maritaca
```

## Setup

Set your Maritaca API key as an environment variable:

```bash
export MARITACA_API_KEY="your-api-key"
```

Or pass it directly to the model:

```python
from langchain_maritaca import ChatMaritaca

model = ChatMaritaca(api_key="your-api-key")
```

## Usage

### Basic Usage

```python
from langchain_maritaca import ChatMaritaca

model = ChatMaritaca(
    model="sabia-3",
    temperature=0.7,
)

messages = [
    ("system", "Você é um assistente prestativo especializado em cultura brasileira."),
    ("human", "Quais são as principais festas populares do Brasil?"),
]

response = model.invoke(messages)
print(response.content)
```

### Streaming

```python
from langchain_maritaca import ChatMaritaca

model = ChatMaritaca(model="sabia-3", streaming=True)

for chunk in model.stream("Conte uma história sobre o folclore brasileiro"):
    print(chunk.content, end="", flush=True)
```

### Async Usage

```python
import asyncio
from langchain_maritaca import ChatMaritaca

async def main():
    model = ChatMaritaca(model="sabia-3")
    response = await model.ainvoke("Qual é a receita de pão de queijo?")
    print(response.content)

asyncio.run(main())
```

### With LangChain Expression Language (LCEL)

```python
from langchain_maritaca import ChatMaritaca
from langchain_core.prompts import ChatPromptTemplate

model = ChatMaritaca(model="sabia-3")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um especialista em {topic}."),
    ("human", "{question}"),
])

chain = prompt | model

response = chain.invoke({
    "topic": "história do Brasil",
    "question": "Quem foi Tiradentes?"
})
print(response.content)
```

## Why Maritaca AI?

Maritaca AI models are specifically trained for Brazilian Portuguese, offering:

- **Native Portuguese Understanding**: Better comprehension of Brazilian idioms, expressions, and cultural context
- **Local Data Training**: Trained on diverse Brazilian Portuguese data sources
- **Cost-Effective**: Competitive pricing for Portuguese language tasks
- **Low Latency**: Servers located in Brazil for faster response times

## API Reference

### ChatMaritaca

Main class for interacting with Maritaca AI models.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"sabia-3"` | Model name to use |
| `temperature` | float | `0.7` | Sampling temperature (0.0-2.0) |
| `max_tokens` | int | None | Maximum tokens to generate |
| `top_p` | float | `0.9` | Top-p sampling parameter |
| `api_key` | str | None | Maritaca API key (or use env var) |
| `base_url` | str | `"https://chat.maritaca.ai/api"` | API base URL |
| `timeout` | float | `60.0` | Request timeout in seconds |
| `max_retries` | int | `2` | Maximum retry attempts |
| `streaming` | bool | `False` | Enable streaming responses |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
