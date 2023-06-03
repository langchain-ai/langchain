# Google Cloud Storage

>[Google Cloud Storage](https://en.wikipedia.org/wiki/Google_Cloud_Storage) is a managed service for storing unstructured data.

## Installation and Setup

First, you need to install `google-cloud-bigquery` python package.

```bash
pip install google-cloud-storage
```

## Document Loader

There are two loaders for the `Google Cloud Storage`: the `Directory` and the `File` loaders.

See a [usage example](../modules/indexes/document_loaders/examples/google_cloud_storage_directory.ipynb).

```python
from langchain.document_loaders import GCSDirectoryLoader
```
See a [usage example](../modules/indexes/document_loaders/examples/google_cloud_storage_file.ipynb).

```python
from langchain.document_loaders import GCSFileLoader
```
