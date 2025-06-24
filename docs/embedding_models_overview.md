# Embedding Models Overview

This brief summary lists common embeddings for Retrieval-Augmented Generation (RAG) systems. The examples in this repository default to the MiniLM model from HuggingFace, but other models may be used depending on use case.

- **HuggingFace Transformer models** – wide variety of open-source models for general text. See the LangChain docs for usage examples.
- **OpenAI embeddings** – powerful embeddings accessible via API for production workloads.
- **Cohere embeddings** – suited for multilingual and long context data.
- **FakeEmbeddings** – lightweight model from LangChain core for testing without network downloads.

Refer to the documentation notebooks under `docs/docs/how_to/indexing.ipynb` and the integration guides in `docs/docs/integrations/` for detailed tutorials.
