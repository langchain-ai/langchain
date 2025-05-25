# Bahasa Indonesia RAG Starter (BM25 + Vector Search)

## Overview
This RAG (Retrieval Augmented Generation) starter demonstrates how to build a system for Bahasa Indonesia that uses hybrid retrieval. It combines BM25 (a keyword-based sparse retrieval method) with dense vector search (using a multilingual sentence transformer model for semantic similarity). The retrieved documents are then used by a Large Language Model (LLM) to generate answers to questions. This notebook is useful for users looking to build RAG applications specifically tailored for Bahasa Indonesia.

## Features
- Hybrid retrieval using BM25 (sparse) and a multilingual sentence transformer model (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`) for dense retrieval.
- Use of LangChain's `EnsembleRetriever` to combine scores from both retrieval methods for more relevant results.
- A complete RAG pipeline for question answering in Bahasa Indonesia.
- Example code and detailed explanations within the `rag_bm25_vector_indonesian.ipynb` notebook.

## Notebook
- `rag_bm25_vector_indonesian.ipynb`: Contains the step-by-step implementation of the Indonesian RAG pipeline.

## Setup and Dependencies
The main libraries required to run the notebook are:
- `langchain`
- `langchain-community`
- `langchain-core`
- `rank_bm25`
- `sentence-transformers`
- `faiss-cpu` (for vector storage)
- `langchain-openai` (for the LLM)
- `tiktoken` (tokenizer for OpenAI models)

You can install them using pip:
```bash
pip install -qU langchain langchain-community langchain-core rank_bm25 sentence-transformers faiss-cpu langchain-openai tiktoken
```

**API Keys:**
Ensure you have your `OPENAI_API_KEY` set as an environment variable if you are using the OpenAI LLM as demonstrated in the notebook. The notebook includes instructions on how to set this up and handles cases where the key might not be present for easier review.

## How to Run
1.  **Install Dependencies:** Ensure all the libraries listed above are installed in your Python environment.
2.  **Set API Keys:** If you plan to use an LLM that requires an API key (like OpenAI's GPT models), make sure the respective API key is correctly set up in your environment (e.g., `OPENAI_API_KEY`).
3.  **Run the Notebook:** Open and execute the cells in the `rag_bm25_vector_indonesian.ipynb` notebook within a Jupyter environment (like Jupyter Lab or Google Colab).

## Customization
This notebook is designed as a starter and can be customized for your specific needs:
-   **Dataset:** Replace the sample Indonesian documents with your own dataset. You'll need to prepare your data in a similar format (list of LangChain `Document` objects).
-   **Embedding Model:** While `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` is a good starting point for multilingual tasks including Bahasa Indonesia, you can experiment with other models from Hugging Face Sentence Transformers or other embedding providers.
-   **LLM:** The notebook uses `ChatOpenAI` by default. You can easily swap this out for other LLMs supported by LangChain (e.g., models from Hugging Face Hub, other cloud providers, or local models).
-   **Retriever Weights:** The `EnsembleRetriever` uses weights to balance the influence of the BM25 and vector retrievers. You can adjust these weights (e.g., `weights=[0.5, 0.5]`) based on empirical results with your dataset and queries to optimize retrieval quality.
-   **Vector Store:** FAISS is used in the notebook. You can replace it with other vector stores supported by LangChain, such as ChromaDB, Pinecone, Weaviate, etc., depending on your requirements.

## Contribution
Contributions and improvements to this notebook are welcome! If you have suggestions for enhancing the RAG pipeline for Bahasa Indonesia, feel free to open an issue or submit a pull request.
