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
