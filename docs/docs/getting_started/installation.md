# Installation

## Official release

To install LangChain run:

```bash
pip install langchain
```

That will install the bare minimum requirements of LangChain.
A lot of the value of LangChain comes when integrating it with various model providers, datastores, etc.
By default, the dependencies needed to do that are NOT installed.
However, there are two other ways to install LangChain that do bring in those dependencies.

To install modules needed for the common LLM providers, run:

```bash
pip install langchain[llms]
```

To install all modules needed for all integrations, run:

```bash
pip install langchain[all]
```

Note that if you are using `zsh`, you'll need to quote square brackets when passing them as an argument to a command, for example:

```bash
pip install 'langchain[all]'
```

## From source

If you want to install from source, you can do so by cloning the repo and running:

```bash
pip install -e .
```
