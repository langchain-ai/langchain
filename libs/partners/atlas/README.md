# langchain-atlas

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-atlas?label=%20)](https://pypi.org/project/langchain-atlas/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-atlas)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

## Quick Install

```bash
pip install langchain-atlas
```

## 🤔 What is this?

<p align="center">
  <img src="./assets/atlas-cloud-logo.png" alt="Atlas Cloud logo" width="240">
</p>

This package contains the LangChain integration with [Atlas Cloud](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=langchain), an OpenAI-compatible MaaS platform for chat models.

🎁 Atlas Cloud is a full-modal AI inference platform that gives developers a single AI API to access video generation, image generation, and LLM APIs. Instead of managing multiple vendor integrations, you connect once and get unified access to 300+ curated models across all modalities.

Check out Atlas Cloud's new coding plan promotion for more budget-friendly API access: [https://www.atlascloud.ai/console/coding-plan](https://www.atlascloud.ai/console/coding-plan)

Validated Atlas 50-LLM chat model pool for `ChatAtlas(model=...)`:

- `deepseek-ai/DeepSeek-V3-0324`, `deepseek-ai/deepseek-r1-0528`, `moonshotai/Kimi-K2-Instruct`, `Qwen/Qwen3-Coder`, `Qwen/Qwen3-235B-A22B-Instruct-2507`
- `deepseek-ai/DeepSeek-V3.1`, `moonshotai/Kimi-K2-Instruct-0905`, `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3-Next-80B-A3B-Thinking`, `Qwen/Qwen3-30B-A3B-Instruct-2507`
- `deepseek-ai/DeepSeek-V3.1-Terminus`, `deepseek-ai/DeepSeek-V3.2-Exp`, `zai-org/GLM-4.6`, `MiniMaxAI/MiniMax-M2`, `Qwen/Qwen3-VL-235B-A22B-Instruct`
- `moonshotai/Kimi-K2-Thinking`, `google/gemini-2.5-flash`, `google/gemini-2.5-flash-lite`, `openai/gpt-5.1`, `openai/gpt-5.1-chat`
- `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-4.1`, `openai/gpt-4.1-mini`, `openai/gpt-4.1-nano`
- `openai/o1`, `openai/o3`, `openai/o3-mini`, `openai/o4-mini`, `anthropic/claude-sonnet-4.5-20250929`
- `deepseek-ai/deepseek-v3.2`, `openai/gpt-5`, `openai/gpt-5-chat`, `openai/gpt-5-mini`, `openai/gpt-5-nano`
- `openai/gpt-5.2`, `openai/gpt-5.2-chat`, `google/gemini-2.5-pro`, `anthropic/claude-opus-4.5-20251101`, `google/gemini-3-flash-preview`
- `zai-org/glm-4.7`, `minimaxai/minimax-m2.1`, `google/gemini-2.0-flash`, `qwen/qwen3-8b`, `qwen/qwen3-235b-a22b-thinking-2507`
- `qwen/qwen3-vl-235b-a22b-thinking`, `qwen/qwen3-30b-a3b`, `qwen/qwen3-30b-a3b-thinking-2507`, `deepseek-ai/deepseek-ocr`, `xai/grok-4-0709`

## Quickstart

```python
from langchain_atlas import ChatAtlas

llm = ChatAtlas(model="deepseek-ai/DeepSeek-V3-0324", temperature=0)
response = llm.invoke("Say hello in Chinese.")
print(response.content)
```

## 📖 Documentation

- Atlas Cloud docs: [https://www.atlascloud.ai/docs](https://www.atlascloud.ai/docs)
- Atlas Cloud LLM docs: [https://www.atlascloud.ai/docs/models/llm](https://www.atlascloud.ai/docs/models/llm)

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
