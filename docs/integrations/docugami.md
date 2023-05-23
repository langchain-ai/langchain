# Docugami

This page covers how to use [Docugami](https://docugami.com) within LangChain.

## What is Docugami?

Docugami converts business documents into a Document XML Knowledge Graph, generating forests of XML semantic trees representing entire documents. This is a rich representation that includes the semantic and structural characteristics of various chunks in the document as an XML tree.

## Quick start

1. Create a Docugami workspace: <a href="http://www.docugami.com">http://www.docugami.com</a> (free trials available)
2. Add your documents (PDF, DOCX or DOC) and allow Docugami to ingest and cluster them into sets of similar documents, e.g. NDAs, Lease Agreements, and Service Agreements. There is no fixed set of document types supported by the system, the clusters created depend on your particular documents, and you can [change the docset assignments](https://help.docugami.com/home/working-with-the-doc-sets-view) later.
3. Create an access token via the Developer Playground for your workspace. Detailed instructions: https://help.docugami.com/home/docugami-api
4. Explore the Docugami API at <a href="https://api-docs.docugami.com">https://api-docs.docugami.com</a> to get a list of your processed docset IDs, or just the document IDs for a particular docset. 
6. Use the DocugamiLoader as detailed in [this notebook](../modules/indexes/document_loaders/examples/docugami.ipynb), to get rich semantic chunks for your documents.
7. Optionally, build and publish one or more [reports or abstracts](https://help.docugami.com/home/reports). This helps Docugami improve the semantic XML with better tags based on your preferences, which are then added to the DocugamiLoader output as metadata. Use techniques like [self-querying retriever](https://python.langchain.com/en/latest/modules/indexes/retrievers/examples/self_query_retriever.html) to do high accuracy Document QA.

# Advantages vs Other Chunking Techniques

Appropriate chunking of your documents is critical for retrieval from documents. Many chunking techniques exist, including simple ones that rely on whitespace and recursive chunk splitting based on character length. Docugami offers a different approach:

1.	**Intelligent Chunking:** Docugami breaks down every document into a hierarchical semantic XML tree of chunks of varying sizes, from single words or numerical values to entire sections. These chunks follow the semantic contours of the document, providing a more meaningful representation than arbitrary length or simple whitespace-based chunking.
2.	**Structured Representation:** In addition, the XML tree indicates the structural contours of every document, using attributes denoting headings, paragraphs, lists, tables, and other common elements, and does that consistently across all supported document formats, such as scanned PDFs or DOCX files. It appropriately handles long-form document characteristics like page headers/footers or multi-column flows for clean text extraction.
3.	**Semantic Annotations:** Chunks are annotated with semantic tags that are coherent across the document set, facilitating consistent hierarchical queries across multiple documents, even if they are written and formatted differently. For example, in set of lease agreements, you can easily identify key provisions like the Landlord, Tenant, or Renewal Date, as well as more complex information such as the wording of any sub-lease provision or whether a specific jurisdiction has an exception section within a Termination Clause.
4.	**Additional Metadata:** Chunks are also annotated with additional metadata, if a user has been using Docugami. This additional metadata can be used for high-accuracy Document QA without context window restrictions. See detailed code walk-through in [this notebook](../modules/indexes/document_loaders/examples/docugami.ipynb).
