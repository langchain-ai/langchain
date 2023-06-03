# Notion DB

>[Notion](https://www.notion.so/) is a collaboration platform with modified Markdown support that integrates kanban 
> boards, tasks, wikis and databases. It is an all-in-one workspace for notetaking, knowledge and data management, 
> and project and task management.

## Installation and Setup

All instructions are in examples below.

## Document Loader

We have two different loaders: `NotionDirectoryLoader` and `NotionDBLoader`.

See a [usage example for the NotionDirectoryLoader](../modules/indexes/document_loaders/examples/notion.ipynb).


```python
from langchain.document_loaders import NotionDirectoryLoader
```

See a [usage example for the NotionDBLoader](../modules/indexes/document_loaders/examples/notiondb.ipynb).


```python
from langchain.document_loaders import NotionDBLoader
```
