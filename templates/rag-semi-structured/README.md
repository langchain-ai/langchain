# Semi structured RAG

This template performs RAG on semi-structured data (e.g., a PDF with text and tables).

See this [blog post](https://langchain-blog.ghost.io/ghost/#/editor/post/652dc74e0633850001e977d4) for useful background context.

## Data loading 

We use [partition_pdf](https://unstructured-io.github.io/unstructured/bricks/partition.html#partition-pdf) from Unstructured to extract both table and text elements.

This will require some system-level package installations, e.g., on Mac:

```
brew install tesseract poppler
```

##  Chroma

[Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma) is an open-source vector database.

This template will create and add documents to the vector database in `chain.py`.

These documents can be loaded from [many sources](https://python.langchain.com/docs/integrations/document_loaders).

##  LLM

Be sure that `OPENAI_API_KEY` is set in order to the OpenAI models.

## Adding the template

Create your LangServe app:
```
langchain serve new my-app
cd my-app
```

Add template:
```
langchain serve add rag-semi-structured
```

Start server:
```
langchain start
```

See Jupyter notebook `rag_semi_structured` for various way to connect to the template.