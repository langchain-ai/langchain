# Extraction with Anthropic Function Calling

This template enables [Anthropic function calling](https://python.langchain.com/docs/integrations/chat/extraction_anthropic_functions).
This is a wrapper around Anthropic's API that uses prompting and output parsing to replicate the OpenAI functions experience.

Specify the information you want to extract in `chain.py`

By default, it will extract the title and author of papers.

##  LLM

This template will use `Claude2` by default. 

Be sure that `ANTHROPIC_API_KEY` is set in your enviorment.
