# Unstructured

This page covers how to use the [`unstructured`](https://github.com/Unstructured-IO/unstructured)
ecosystem within LangChain. The `unstructured` package from
[Unstructured.IO](https://www.unstructured.io/) extracts clean text from raw source documents like
PDFs and Word documents.


This page is broken into two parts: installation and setup, and then references to specific
`unstructured` wrappers.

## Installation and Setup
- Install the Python SDK with `pip install "unstructured[local-inference]"`
- Install the following system dependencies if they are not already available on your system.
  Depending on what document types you're parsing, you may not need all of these.
    - `libmagic-dev`
    - `poppler-utils`
    - `tesseract-ocr`
    - `libreoffice`
- If you are parsing PDFs, run the following to install the `detectron2` model, which
  `unstructured` uses for layout detection:
    - `pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2"`

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
