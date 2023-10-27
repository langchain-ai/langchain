# Summarize documents with Anthropic

This template uses Anthropic's `Claude2` to summarize documents.

To do this, we can use various prompts from LangChain hub, such as:

* [This fun summarization prompt](https://smith.langchain.com/hub/hwchase17/anthropic-paper-qa)
* [Chain of density summarization prompt](https://smith.langchain.com/hub/lawwu/chain_of_density)

`Claude2` has a large (100k token) context window, allowing us to summarize documents over 100 pages.

##  LLM

This template will use `Claude2` by default. 

Be sure that `ANTHROPIC_API_KEY` is set in your enviorment.
