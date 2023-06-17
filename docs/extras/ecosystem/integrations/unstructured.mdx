# Unstructured

>The `unstructured` package from
[Unstructured.IO](https://www.unstructured.io/) extracts clean text from raw source documents like
PDFs and Word documents.
This page covers how to use the [`unstructured`](https://github.com/Unstructured-IO/unstructured)
ecosystem within LangChain.

## Installation and Setup

If you are using a loader that runs locally, use the following steps to get `unstructured` and
its dependencies running locally.

- Install the Python SDK with `pip install "unstructured[local-inference]"`
- Install the following system dependencies if they are not already available on your system.
  Depending on what document types you're parsing, you may not need all of these.
    - `libmagic-dev` (filetype detection)
    - `poppler-utils` (images and PDFs)
    - `tesseract-ocr`(images and PDFs)
    - `libreoffice` (MS Office docs)
    - `pandoc` (EPUBs)

If you want to get up and running with less set up, you can
simply run `pip install unstructured` and use `UnstructuredAPIFileLoader` or
`UnstructuredAPIFileIOLoader`. That will process your document using the hosted Unstructured API.
Note that currently (as of 1 May 2023) the Unstructured API is open, but it will soon require
an API. The [Unstructured documentation page](https://unstructured-io.github.io/) will have
instructions on how to generate an API key once they're available. Check out the instructions
[here](https://github.com/Unstructured-IO/unstructured-api#dizzy-instructions-for-using-the-docker-image)
if you'd like to self-host the Unstructured API or run it locally.

## Wrappers

### Data Loaders

The primary `unstructured` wrappers within `langchain` are data loaders. The following
shows how to use the most basic unstructured data loader. There are other file-specific
data loaders available in the `langchain.document_loaders` module.

```python
from langchain.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader("state_of_the_union.txt")
loader.load()
```

If you instantiate the loader with `UnstructuredFileLoader(mode="elements")`, the loader
will track additional metadata like the page number and text type (i.e. title, narrative text)
when that information is available.
