# Installation Options

LangChain is available on PyPi, so to it is easily installable with:

```
pip install langchain
```

That will install the bare minimum requirements of LangChain.
A lot of the value of LangChain comes when integrating it with various model providers, datastores, etc.
By default, the dependencies needed to do that are NOT installed.
However, there are two other ways to install LangChain that do bring in those dependencies.

To install modules needed for the common LLM providers, run:

```
pip install langchain[llms]
```

To install all modules needed for all integrations, run:

```
pip install langchain[all]
```