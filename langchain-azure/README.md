# 🦜️🔗 LangChain 💜 Azure

This repository provides LangChain components for various Azure services.

Usage

This repository contains five packages with Azure integrations within LangChain:
- [langchain-azure-ai]() implements integrations of Azure AI Services, Cognitive Services, Azure Search and AzureOpenAI 
- [langchain-azure-ml]() implements integrations of [Azure ML](https://learn.microsoft.com/en-gb/python/api/overview/azure/ai-ml-readme?view=azure-python), which primarily inlcudes [Azure ML Endpoint](https://python.langchain.com/docs/integrations/llms/azure_ml/) at the moment. 
- [langchain-azure-cosmos]() implements integrations of [Azure Cosmos](https://pypi.org/project/azure-cosmos/), which includes the Azure Cosmos DB vector store. 
- [langchain-azure-dynamic-sessions]() implements integrations for Azure Container Apps dynamic sessions. You can use it to add a secure and scalable code interpreter to your agents.
- [langchain-azure-sql]() implements integrations of the new azure-sql package. 

Each of these has its own development environment. Docs are run from the top-level makefile, but development
is split across separate test & release flows.

## Repository Structure

If you plan on contributing to LangChain-Azure code or documentation, it can be useful
to understand the high level structure of the repository.

LangChain-Azure is organized as a [monorepo](https://en.wikipedia.org/wiki/Monorepo) that contains multiple packages.

Here's the structure visualized as a tree:

```text
.
├── libs
│   ├── ai
│   │   ├── tests/unit_tests # Unit tests (present in each package not shown for brevity)
│   │   ├── tests/integration_tests # Integration tests (present in each package not shown for brevity)
│   ├── azure-dynamic-sessions
│   ├── cosmos
│   ├── ml
│   ├── sql
```

The root directory also contains the following files:

* `pyproject.toml`: Dependencies for building docs and linting docs, cookbook.
* `Makefile`: A file that contains shortcuts for building, linting and docs and cookbook.

There are other files in the root directory level, but their presence should be self-explanatory. Feel free to browse around!

