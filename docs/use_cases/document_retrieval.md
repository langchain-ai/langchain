# Document Retrieval

Overview

LangChain can be used for document retrieval stored in a VectorStore. The documents are embedded using one of the multiple LangChain LLMs and then retrieved based on a user query. 

## Document Retrieval by LLMs Tagging

You can also leverage LangChain APIs to generate a set of domain-specific tags for each document and store them instead of the full text. Then, to search, you can translate a user query to tags using LLMs to get better results. A concrete example may be a list of items' descriptions to see to sell (for example, "New pair of shoes", "Used Levis jeans") and user input describing what he/she is looking for (for example, "I need new gym clothes"). You can prompt the LLMs asking for more or less refined tags based on your domain. We can first translate each text to a fixed set of tags, "shoes", and "jeans", and then do the retrieval by first translating the user query to tags, "shoes, shorts", and then do the similarity search in the vector db. Effectively restraining the search space to yield better matches.

LangChain Workflow for Document Retrieval with shifted domain

1. Index the documents: Given a set of documents, translate each document into a set of tags using a LLMs with a custom prompt template

2. Embedding and Storage: Store the resulting tags in a Vector Storage

3. Translate User Query: Use one of LangChain LLMs to translate the user query into tags.

4. Retrieve: Use LangChain Vector Store to retrieve the documents matching the user query translated into tags 


The full tutorial is available below.
- [FairytaleDJ](https://www.activeloop.ai/resources/3-ways-to-build-a-recommendation-engine-for-songs-with-lang-chain/): An application to recommend Disney songs based on user feellings/moods and vibes. An interactive demo can be found on [Hugging Face Spaces](https://huggingface.co/spaces/Francesco/FairytaleDJ)


