# Docugami

This page covers how to use [Docugami](https://docugami.com) within LangChain.

## What is Docugami?

Docugami converts business documents, into a Document XML Knowledge Graph, generating forests of XML semantic trees representing entire documents. This is a rich representation that includes the semantic and structural characteristics of various chunks in the document as an XML tree.

## Quick start

1. Create a Docugami workspace: http://www.docugami.com (free trials available)
2. Add your documents (PDF, DOCX, DOC) and allow Docugami to ingest and cluster them into sets of similar documents, e.g. NDAs, Lease Agreements, and Service Agreements. There is no fixed set of documents supported by the system, the clusters created depend on your particular documents, and you can [change the docset assignments](https://help.docugami.com/home/working-with-the-doc-sets-view) later.
3. Build and publish one or more reports or abstracts. This helps Docugami improve the semantic XML with better tags based on your preferences.
4. Create an access token via the Developer Playground for your workspace. Detailed instructions: https://help.docugami.com/home/docugami-api
5. Explore the Docugami API at https://api-docs.docugami.com/ to get a list of your processed docset IDs, or just the document IDs for a particular docset. 
6. Use the DocugamiLoader as detailed in [this notebook](../modules/indexes/document_loaders/examples/docugami.ipynb), to get rich semantic chunks for your documents.

# Advantages vs Other Chunking Techniques

Appropriate chunking of your documents is critical for retrieval from documents. Many chunking techniques exist, including simple ones that rely on whitespace and recursive chunk splitting based on character length. Docugami offers a different approach:

1. The XML tree hierarchically represents the information that is in headings, paragraphs, lists, tables, and other common elements, consistently across all supported document formats, such as scanned PDFs or docx. 
2. Every document is broken down into chunks of varying sizes, from a single word or numerical value to an address to an entire section or more.
3. Chunks are annotated with semantic tags that are coherent across the document set, for example facilitating consistent hierarchical queries across a set of leases, such as identifying the Landlord, a Tenant or if a specific Country has an exception section inside of a Termination Clause.