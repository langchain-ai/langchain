# Calling a LLM

The most basic building block of LangChain is calling an LLM on some input.
Let's walk through a simple example of how to do this. 
For this purpose, let's pretend we are building a service that generates a company name based on what the company makes.

In order to do this, we first need to import the LLM wrapper.

```python
from langchain.llms import OpenAI
```

We can then initialize the wrapper with any arguments.
In this example, we probably want the outputs to be MORE random, so we'll initialize it with a HIGH temperature.

```python
llm = OpenAI(temperature=0.9)
```

We can now call it on some input!

```python
text = "What would be a good company name a company that makes colorful socks?"
llm(text)
```
