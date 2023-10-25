# Semi-sturctured RAG 

This template enables RAG on PDF documents that contain a mixture of tables and text.

## Data loading 

We use [partition_pdf](https://unstructured-io.github.io/unstructured/bricks/partition.html#partition-pdf) from Unstructured to extract both table and text elements.

This will require some system-level package installations, e.g., on Mac:

```
brew install tesseract poppler
```

## Embedding, LLM

Be sure that `OPENAI_API_KEY` is set in order to the OpenAI model(s) for embedding and answer synthesis.

## Installation

```bash
# from inside your LangServe instance
poe add rag-semi-structured
```