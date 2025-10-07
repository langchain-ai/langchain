# Packages

> [!IMPORTANT]
> [**View all LangChain integrations packages**](https://docs.langchain.com/oss/python/integrations/providers)

This repository is structured as a monorepo, with various packages located in this `libs/` directory. Packages to note in this directory include:

```txt
core/             # Core primitives and abstractions for langchain
langchain/        # langchain-classic
langchain_v1/     # langchain
partners/         # Certain third-party providers integrations (see below)
standard-tests/   # Standardized tests for integrations
text-splitters/   # Text splitter utilities
```

(Each package contains its own `README.md` file with specific details about that package.)

## Integrations (`partners/`)

The `partners/` directory contains a small subset of third-party provider integrations that are maintained directly by the LangChain team. These include, but are not limited to:

* [OpenAI](https://pypi.org/project/langchain-openai/)
* [Anthropic](https://pypi.org/project/langchain-anthropic/)
* [Ollama](https://pypi.org/project/langchain-ollama/)
* [DeepSeek](https://pypi.org/project/langchain-deepseek/)
* [xAI](https://pypi.org/project/langchain-xai/)
* and more

Most integrations have been moved to their own repositories for improved versioning, dependency management, collaboration, and testing. This includes packages from popular providers such as [Google](https://github.com/langchain-ai/langchain-google) and [AWS](https://github.com/langchain-ai/langchain-aws). Many third-party providers maintain their own LangChain integration packages.

For a full list of all LangChain integrations, please refer to the [LangChain Integrations documentation](https://docs.langchain.com/oss/python/integrations/providers).
