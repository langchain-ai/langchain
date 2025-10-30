# 🦜🪪 langchain-model-profiles

> [!WARNING]
> This package is currently in development and the API is subject to change.

Centralized reference of LLM capabilities for LangChain chat models.

## Overview

`langchain-model-profiles` enables programmatic access to model capabilities through a
`.profile` property on LangChain chat models.

This allows you to query model-specific features such as context window sizes, supported
input/output modalities, structured output support, tool calling capabilities, and more.

## Data Sources

This package is built on top of the excellent work by the
[models.dev](https://github.com/sst/models.dev) project, an open source initiative that
provides model capability data.

This package augments the data from models.dev with some additional fields.

## Installation

```bash
pip install "langchain[model-profiles]"
```

Or with uv:

```bash
uv add "langchain[model-profiles]"
```

## Usage

Access model capabilities through the `.profile` property on any LangChain chat model:

```python
from langchain.chat_models import init_chat_model

# Initialize a chat model
model = init_chat_model("openai:gpt-5")

# Access the model profile
profile = model.profile

# Check specific capabilities
if profile.get("structured_output"):
    print(f"This model supports a dedicated structured output feature.")

if profile.get("max_input_tokens"):
    print(f"Max input tokens: {profile.get('max_input_tokens')}")
```

## Available Profile Fields

The `ModelProfile` TypedDict includes the following fields:

> [!WARNING]
> This package is currently in development and these fields are subject to change.

### Input Constraints
- `max_input_tokens` (int): Maximum number of input tokens
- `image_inputs` (bool): Support for image inputs
- `image_url_inputs` (bool): Support for image URL inputs
- `pdf_inputs` (bool): Support for PDF inputs
- `audio_inputs` (bool): Support for audio inputs
- `video_inputs` (bool): Support for video inputs
- `image_tool_message` (bool): Support for images in tool messages
- `pdf_tool_message` (bool): Support for PDFs in tool messages

### Output Constraints
- `max_output_tokens` (int): Maximum number of output tokens
- `reasoning_output` (bool): Support for reasoning/thinking tokens
- `image_outputs` (bool): Can generate image outputs
- `audio_outputs` (bool): Can generate audio outputs
- `video_outputs` (bool): Can generate video outputs

### Tool Calling
- `tool_calling` (bool): Supports tool/function calling
- `tool_choice` (bool): Supports forcing specific tool calls

### Structured Output
- `structured_output` (bool): Supports dedicated structured output features

## Development

```bash
# Install dependencies
uv sync

# Format code
make format

# Run linting
make lint

# Run tests
make test

```

## License

MIT
