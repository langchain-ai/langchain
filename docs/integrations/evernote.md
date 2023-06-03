# EverNote

>[EverNote](https://evernote.com/) is intended for archiving and creating notes in which photos, audio and saved web content can be embedded. Notes are stored in virtual "notebooks" and can be tagged, annotated, edited, searched, and exported.

## Installation and Setup

First, you need to install `lxml` and `html2text` python packages.

```bash
pip install lxml
pip install html2text
```

## Document Loader

See a [usage example](../modules/indexes/document_loaders/examples/evernote.ipynb).

```python
from langchain.document_loaders import EverNoteLoader
```
