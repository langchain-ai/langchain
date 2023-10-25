# Conversational RAG 

[Conversational](https://python.langchain.com/docs/expression_language/cookbook/retrieval#conversational-retrieval-chain) [retrieval](https://python.langchain.com/docs/use_cases/question_answering/) is one of the most popular LLM use-cases.

It passes both a conversation history and retrieved documents into an LLM for synthesis.

`Add template`

* When we add a template, we update our LangServe app's Poetry config file with the necessary dependencies.
* It also automatically installs these template dependencies in your Poetry environment.
```
langchain serve add rag-conversation
```

`Start FastAPI server`
```
langchain start
```

See the notebook for various ways to interact with this template.