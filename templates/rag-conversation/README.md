# Conversational RAG

This template performs [conversational](https://python.langchain.com/docs/expression_language/cookbook/retrieval#conversational-retrieval-chain) [retrieval](https://python.langchain.com/docs/use_cases/question_answering/), which is one of the most popular LLM use-cases.

It passes both a conversation history and retrieved documents into an LLM for synthesis.

##  LLM

Be sure that `OPENAI_API_KEY` is set in order to use the OpenAI models.

##  Pinecone

This template uses Pinecone as a vectorstore and requires that `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, and `PINECONE_INDEX` are set.