# langchain-anthropic

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-anthropic?label=%20)](https://pypi.org/project/langchain-anthropic/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-anthropic)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-anthropic)](https://pypistats.org/packages/langchain-anthropic)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
uv add langchain-anthropic
```

## 🤔 What is this?

This package contains the LangChain integration for Anthropic's generative models.

## 📖 Documentation

For full documentation, see the [API reference](https://reference.langchain.com/python/integrations/langchain_anthropic/). For conceptual guides, tutorials, and examples on using these classes, see the [LangChain Docs](https://docs.langchain.com/oss/python/integrations/providers/anthropic).

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).

## Migrating to Claude Sonnet 5.5

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
    model="claude-sonnet-5-5",
    max_tokens=16000,
    output_config={"effort": "medium"},
)
```

- Use `with_structured_output(schema, method="json_schema")` for native structured output. Sonnet 5.5 rejects forced tool choice (`"any"` or a tool name). Function-calling structured output does not force a call and raises a parsing error if the model answers without one.
- Thinking is adaptive by default. For no up-front thinking, use `thinking={"type": "between_tools"}` at `high` effort or below, with no additional thinking fields. `disabled` and budgeted `enabled` thinking are unsupported.
- Omit sampling settings; non-default `temperature`, `top_p`, and `top_k` are rejected. Budget output tokens for both thinking and text.
- Preserve signed thinking blocks, including empty ones, and keep history append-only. Use mid-conversation system messages to change instructions or tools rather than editing earlier turns.
- Progress updates can arrive as thinking blocks. Use adaptive thinking with `display="summarized"` or `display="updates"` to display them; the latter's beta header is added automatically.
- Computer use on the direct Claude API requires `computer_toolset_20260801`. Preserve the returned tool-use content: its `toolset_name` is retained on replay and copied to matching tool results. Remove the old fine-grained streaming beta when using toolsets.

See the [migration guide](https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide) for platform-specific restrictions, advisor pairings, and refusal/fallback behavior.

## Resources

- [LangChain Academy](https://academy.langchain.com/) — comprehensive, free courses on LangChain libraries and products, made by the LangChain team
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) — community guidelines and standards
