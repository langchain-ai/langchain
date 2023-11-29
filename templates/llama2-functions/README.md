# Extraction with LLaMA2 Function Calling

This template shows how to do extraction of structured data from unstructured data, using LLaMA2 [fine-tuned for grammars and jsonschema](https://replicate.com/andreasjansson/llama-2-13b-chat-gguf).

Specify the scehma you want to extract in `chain.py`

By default, it will extract the title and author of papers.

##  LLM

This template will use a `Replicate` [hosted version](https://replicate.com/andreasjansson/llama-2-13b-chat-gguf) of LLaMA2 that has support for grammars and jsonschema. 

Based on the `Replicate` example, these are supplied directly in the prompt.

Be sure that `REPLICATE_API_TOKEN` is set in your environment.