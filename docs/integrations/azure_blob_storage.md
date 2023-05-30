# Azure Blob Storage

>[Azure Blob Storage](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blobs-introduction) is Microsoft's object storage solution for the cloud. Blob Storage is optimized for storing massive amounts of unstructured data. Unstructured data is data that doesn't adhere to a particular data model or definition, such as text or binary data.

>[Azure Files](https://learn.microsoft.com/en-us/azure/storage/files/storage-files-introduction) offers fully managed
> file shares in the cloud that are accessible via the industry standard Server Message Block (`SMB`) protocol, 
> Network File System (`NFS`) protocol, and `Azure Files REST API`. `Azure Files` are based on the `Azure Blob Storage`.

`Azure Blob Storage` is designed for:
- Serving images or documents directly to a browser.
- Storing files for distributed access.
- Streaming video and audio.
- Writing to log files.
- Storing data for backup and restore, disaster recovery, and archiving.
- Storing data for analysis by an on-premises or Azure-hosted service.

## Installation and Setup

```bash
pip install azure-storage-blob
```


## Document Loader

See a [usage example for the Azure Blob Storage](../modules/indexes/document_loaders/examples/azure_blob_storage_container.ipynb).

```python
from langchain.document_loaders import AzureBlobStorageContainerLoader
```

See a [usage example for the Azure Files](../modules/indexes/document_loaders/examples/azure_blob_storage_file.ipynb).

```python
from langchain.document_loaders import AzureBlobStorageFileLoader
```
