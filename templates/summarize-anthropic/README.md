# Summarize PDFs with Anthropic

We can use Claude2, which has a long context window, to summarize PDFs.

To do this, we can use various prompts from LangChain hub, such as:

* [This fun summarization prompt](https://smith.langchain.com/hub/hwchase17/anthropic-paper-qa)
* [Chain of density summarization prompt](https://smith.langchain.com/hub/lawwu/chain_of_density)

`Claude2` has a large (100k token) context window, allowing us to summarize documents over 100 pages.

`Add template`

* When we add a template, we update our LangServe app's Poetry config file with the necessary dependencies.
* It also automatically installs these template dependencies in your Poetry environment.
```
langchain serve add summarize-anthropic
```

`Start FastAPI server`
```
langchain start
```