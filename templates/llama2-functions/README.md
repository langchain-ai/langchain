# Extraction with LLaMA2 Function Calling

This template shows how to do extraction of structured data from unstructured data, using LLaMA2 [fine-tuned for grammars and jsonschema](https://replicate.com/andreasjansson/llama-2-13b-chat-gguf).

[Query transformations](https://blog.langchain.dev/query-transformations/) are one great application area for open source, private LLMs:

* The tasks are often narrow and well-defined (e.g., generatae multiple questions from a user input)
* They also are tasks that users may want to run locally (e.g., in a RAG workflow)

Specify the scehma you want to extract in `chain.py`

##  LLM

This template will use a `Replicate` [hosted version](https://replicate.com/andreasjansson/llama-2-13b-chat-gguf) of LLaMA2 that has support for grammars and jsonschema. 

Based on the `Replicate` example, the JSON schema is supplied directly in the prompt.

Be sure that `REPLICATE_API_TOKEN` is set in your environment.