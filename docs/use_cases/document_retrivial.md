# Document Retrivial

Overview

LangChain can be use for document retrivial stored in a VectorStore. The documents are embedded using one of the multiple LangChain LLMs and then retrieved based on an user query. 

## Document Retrivial with shifted domain

You can also leverage LangChain APIs to generated a set of domain specific tokens for each document, e.g. tags, store them and then search for similarity by translating an user query first to the correct tags and then perform the search. A concrete example may be a list of items's description to see to sell (example, "New pair of shoes", "Used levis jeans") and user input describing what he/she is looking for (example, "I need a new t-shirt"). We can first translate each text to a fixed set of tags, "shoes", "jeans", and the do the retrivial based on similiraty on the tags effectevely restrain the search space to yield better matches.

LangChain Workflow for Document Retrivial with shifted domain

1. Index the documents base: Given a set of documents, translate each document into a set of tags using a LLMs with a custom prompt template

2. Embedding and Storaged: Store the resulting tags into a Vector Storage

3. Translate User Query: Use one of LangChain LLMs to transalte the user query into tags.

4. Retrieve: Use LangChain Vector Store to retrieve the documents matching the user query translated into tags 


The full tutorial is available below.
- [TODO](TODO): An application to reccomand Disney songs based on user feellings/moods and vibes.

An interactive demo can be found on [Hugging Face Spaces](TODO)


