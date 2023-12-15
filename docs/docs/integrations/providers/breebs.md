# BREEBS

[BREEBS](https://www.breebs.com/) is a collaborative knowledge platform. 
Anybody can create a Breeb, a knowledge capsule, based on PDFs stored on a Google Drive folder.
A breeb can be used by any LLM/chatbot to improve its expertise, reduce hallucinations and give access to sources.

## List of available Breebs

To get the full list of Breebs, including their key (breeb_key) and description : 
https://breebs.promptbreeders.com/web/listbreebs.


## Installation and Setup
```python
pip install langchain
```
- Get a breeb key.

## Retriever
```python
from langchain.retrievers import BreebsRetriever
```

# Example
[Integration with langchain](https://python.langchain.com/docs/integrations/retrievers/breebs)