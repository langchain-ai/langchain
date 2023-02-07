# Key Concepts

## Document
This class is a container for document information. This contains two parts:
- `page_content`: The content of the actual page itself.
- `metadata`: The metadata associated with the document. This can be things like the file path, the url, etc.

## Loader
This base class is a way to load documents. It exposes a `load` method that returns `Document` objects.

## [Unstructured](https://github.com/Unstructured-IO/unstructured)
Unstructured is a python package specifically focused on transformations from raw documents to text.
