# Installation

## Official Releases

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

Note that if you are using `zsh`, you'll need to quote square brackets when passing them as an argument to a command, for example:

```
pip install 'langchain[all]'
```

## Installing from source

If you want to install from source, you can do so by cloning the repo and running:

```
pip install -e .
```
