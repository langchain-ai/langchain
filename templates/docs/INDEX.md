# Templates

Highlighting a few different categories of templates

## ⭐ Popular

These are some of the more popular templates to get started with.

- [Retrieval Augmented Generation Chatbot](../rag-conversation): Build a chatbot over your data. Defaults to OpenAI and Pinecone.
- [Extraction with OpenAI Functions](../extraction-openai-functions): Do extraction of structured data from unstructured data. Uses OpenAI function calling.
- [Local Retrieval Augmented Generation](../rag-chroma-private): Build a chatbot over your data. Uses only local tooling: Ollama, GPT4all, Chroma.
- [OpenAI Functions Agent](../openai-functions-agent): Build a chatbot that can take actions. Uses OpenAI function calling and Tavily.
- [XML Agent](../xml-agent): Build a chatbot that can take actions. Uses Anthropic and You.com.


## 📥 Advanced Retrieval

These templates cover advanced retrieval techniques.

- [Reranking](../rag-pinecone-rerank): This retrieval technique uses Cohere's reranking endpoint to rerank documents from an initial retrieval step.
- [Anthropic Iterative Search](../anthropic-iterative-search): This retrieval technique uses iterative prompting to determine what to retrieve and whether the retriever documents are good enough.
- [Neo4j Parent Document Retrieval](../neo4j-parent): This retrieval technique stores embeddings for smaller chunks, but then returns larger chunks to pass to the model for generation.
- [Semi-Structured RAG](../rag-semi-structured): The template shows how to do retrieval over semi-structured data (e.g. data that involves both text and tables).

## 🔍Advanced Retrieval - Query Transformation

A selection of advanced retrieval methods that involve transforming the original user query.

- [Hypothetical Document Embeddings](../hyde): A retrieval technique that generates a hypothetical document for a given query, and then uses the embedding of that document to do semantic search. [Paper](https://arxiv.org/abs/2212.10496).
- [Rewrite-Retrieve-Read](../rewrite-retrieve-read): A retrieval technique that rewrites a given query before passing it to a search engine. [Paper](https://arxiv.org/abs/2305.14283).
- [Step-back QA Prompting](../stepback-qa-prompting): A retrieval technique that generates a "step-back" question and then retrieves documents relevant to both that question and the original question. [Paper](https://arxiv.org/abs//2310.06117).
- [RAG-Fusion](../rag-fusion): A retrieval technique that generates multiple queries and then reranks the retrieved documents using reciprocal rank fusion. [Article](https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1).
- [Multi-Query Retriever](../rag-pinecone-multi-query): This retrieval technique uses an LLM to generate multiple queries and then fetches documents for all queries.


## 🧠Advanced Retrieval - Query Construction

A selection of advanced retrieval methods that involve constructing a query in a separate DSL from natural language.

- [Elastic Query Generator](../elastic-query-generator): Generate elastic search queries from natural language.
- [Neo4j Cypher Generation](../neo4j-cypher): Generate cypher statements from natural language. Available with a ["full text" option](../neo4j-cypher-ft) as well.
- [Supabase Self Query](../self-query-supabase): Parse a natural language query into a semantic query as well as a metadata filter for Supabase.


## 🦙 OSS Models

These templates use OSS models.

- [Local Retrieval Augmented Generation](../rag-chroma-private): Build a chatbot over your data. Uses only local tooling: Ollama, GPT4all, Chroma.
- [SQL Question Answering (Replicate)](../sql-llama2): Question answering over a SQL database, using Llama2 hosted on [Replicate](https://replicate.com/).
- [SQL Question Answering (LlamaCpp)](../sql-llamacpp): Question answering over a SQL database, using Llama2 through [LlamaCpp](https://github.com/ggerganov/llama.cpp).
- [SQL Question Answering (Ollama)](../sql-ollama): Question answering over a SQL database, using Llama2 through [Ollama](https://github.com/jmorganca/ollama).

## ⛏️ Extraction

Extract data in a structured format
- [Extraction Using OpenAI Functions](../extraction-openai-functions): Extract information from text using OpenAI Function Calling.
- [Extraction Using Anthropic Functions](../extraction-anthropic-functions): Extract information from text using a LangChain wrapper around the Anthropic endpoints intended to simulate function calling.
- [Extract BioTech Plate Data](../plate-chain): Extract microplate data from messy Excel spreadsheets into a more normalized format.

## 🤖 Agents
- [OpenAI Functions Agent](../openai-functions-agent): Build a chatbot that can take actions. Uses OpenAI function calling and Tavily.
- [XML Agent](../xml-agent): Build a chatbot that can take actions. Uses Anthropic and You.com.
