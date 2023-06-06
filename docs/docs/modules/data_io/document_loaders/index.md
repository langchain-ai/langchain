# Document Loaders

Combining language models with your own text data is a powerful way to differentiate them.
The first step in doing this is to load the data into "Documents" - a fancy way to say some pieces of text.
The document loader is aimed at making this easy.

The following document loaders are provided:

## Transform loaders

These **transform** loaders transform data from a specific format into the Document format.
For example, there are **transformers** for CSV and SQL.
Mostly, these loaders input data from files but sometime from URLs.

A primary driver of a lot of these transformers is the `Unstructured <https://github.com/Unstructured-IO/unstructured>`_ python package.
This package transforms many types of files - text, powerpoint, images, html, pdf, etc - into text data.

For detailed instructions on how to get set up with Unstructured, see installation guidelines `here <https://github.com/Unstructured-IO/unstructured#coffee-getting-started>`_.


## Public dataset or service loaders
These datasets and sources are created for public domain and we use queries to search there
and download necessary documents.
For example, **Hacker News** service.

We don't need any access permissions to these datasets and services.


## Proprietary dataset or service loaders
These datasets and services are not from the public domain.
These loaders mostly transform data from specific formats of applications or cloud services,
for example **Google Drive**.

We need access tokens and sometime other parameters to get access to these datasets and services.
