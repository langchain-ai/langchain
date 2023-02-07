# Quickstart Guide


This tutorial gives you a quick walkthrough about building an end-to-end language model application with LangChain.

## Installation

To get started, install LangChain with the following command:

```bash
pip install langchain
```


## Environment Setup

Using LangChain will usually require integrations with one or more model providers, data stores, apis, etc.

For this example, we will be using OpenAI's APIs, so we will first need to install their SDK:

```bash
pip install openai
```

We will then need to set the environment variable in the terminal.

```bash
export OPENAI_API_KEY="..."
```

Alternatively, you could do this from inside the Jupyter notebook (or Python script):

```python
import os
os.environ["OPENAI_API_KEY"] = "..."
```


## Building a Language Model Application

Now that we have installed LangChain and set up our environment, we can start building our language model application.

LangChain provides many modules that can be used to build language model applications. Modules can be combined to create more complex applications, or be used individually for simple applications.



`````{dropdown} LLMs: Get predictions from a language model

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
print(llm(text))
```

```pycon
Feetful of Fun
```

For more details on how to use LLMs within LangChain, see the [LLM getting started guide](../modules/llms/getting_started.ipynb).
`````


`````{dropdown} Prompt Templates: Manage prompts for LLMs

Calling an LLM is a great first step, but it's just the beginning.
Normally when you use an LLM in an application, you are not sending user input directly to the LLM.
Instead, you are probably taking user input and constructing a prompt, and then sending that to the LLM.

For example, in the previous example, the text we passed in was hardcoded to ask for a name for a company that made colorful socks.
In this imaginary service, what we would want to do is take only the user input describing what the company does, and then format the prompt with that information.

This is easy to do with LangChain!

First lets define the prompt template:

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
```

Let's now see how this works! We can call the `.format` method to format it.

```python
print(prompt.format(product="colorful socks"))
```

```pycon
What is a good name for a company that makes colorful socks?
```


[For more details, check out the getting started guide for prompts.](../modules/prompts/getting_started.ipynb)

`````



`````{dropdown} Chains: Combine LLMs and prompts in multi-step workflows

Up until now, we've worked with the PromptTemplate and LLM primitives by themselves. But of course, a real application is not just one primitive, but rather a combination of them.

A chain in LangChain is made up of links, which can be either primitives like LLMs or other chains.

The most core type of chain is an LLMChain, which consists of a PromptTemplate and an LLM.

Extending the previous example, we can construct an LLMChain which takes user input, formats it with a PromptTemplate, and then passes the formatted response to an LLM.

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
```

We can now create a very simple chain that will take user input, format the prompt with it, and then send it to the LLM:

```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
```

Now we can run that chain only specifying the product!

```python
chain.run("colorful socks")
# -> '\n\nSocktastic!'
```

There we go! There's the first chain - an LLM Chain.
This is one of the simpler types of chains, but understanding how it works will set you up well for working with more complex chains.

[For more details, check out the getting started guide for chains.](../modules/chains/getting_started.ipynb)

`````


`````{dropdown} Agents: Dynamically call chains based on user input

So far the chains we've looked at run in a predetermined order.

Agents no longer do: they use an LLM to determine which actions to take and in what order. An action can either be using a tool and observing its output, or returning to the user.

When used correctly agents can be extremely powerful. In this tutorial, we show you how to easily use agents through the simplest, highest level API.


In order to load agents, you should understand the following concepts:

- Tool: A function that performs a specific duty. This can be things like: Google Search, Database lookup, Python REPL, other chains. The interface for a tool is currently a function that is expected to have a string as an input, with a string as an output.
- LLM: The language model powering the agent.
- Agent: The agent to use. This should be a string that references a support agent class. Because this notebook focuses on the simplest, highest level API, this only covers using the standard supported agents. If you want to implement a custom agent, see the documentation for custom agents (coming soon).

**Agents**: For a list of supported agents and their specifications, see [here](../modules/agents/agents.md).

**Tools**: For a list of predefined tools and their specifications, see [here](../modules/agents/tools.md).

For this example, you will also need to install the SerpAPI Python package.

```bash
pip install google-search-results
```

And set the appropriate environment variables.

```python
import os
os.environ["SERPAPI_API_KEY"] = "..."
```

Now we can get started!

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

# First, let's load the language model we're going to use to control the agent.
llm = OpenAI(temperature=0)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)


# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Now let's test it out!
agent.run("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")
```

```pycon
Entering new AgentExecutor chain...
 I need to find out who Olivia Wilde's boyfriend is and then calculate his age raised to the 0.23 power.
Action: Search
Action Input: "Olivia Wilde boyfriend"
Observation: Jason Sudeikis
Thought: I need to find out Jason Sudeikis' age
Action: Search
Action Input: "Jason Sudeikis age"
Observation: 47 years
Thought: I need to calculate 47 raised to the 0.23 power
Action: Calculator
Action Input: 47^0.23
Observation: Answer: 2.4242784855673896

Thought: I now know the final answer
Final Answer: Jason Sudeikis, Olivia Wilde's boyfriend, is 47 years old and his age raised to the 0.23 power is 2.4242784855673896.
> Finished AgentExecutor chain.
"Jason Sudeikis, Olivia Wilde's boyfriend, is 47 years old and his age raised to the 0.23 power is 2.4242784855673896."
```


`````


`````{dropdown} Memory: Add state to chains and agents

So far, all the chains and agents we've gone through have been stateless. But often, you may want a chain or agent to have some concept of "memory" so that it may remember information about its previous interactions. The clearest and simple example of this is when designing a chatbot - you want it to remember previous messages so it can use context from that to have a better conversation. This would be a type of "short-term memory". On the more complex side, you could imagine a chain/agent remembering key pieces of information over time - this would be a form of "long-term memory". For more concrete ideas on the latter, see this [awesome paper](https://memprompt.com/).

LangChain provides several specially created chains just for this purpose. This notebook walks through using one of those chains (the `ConversationChain`) with two different types of memory.

By default, the `ConversationChain` has a simple type of memory that remembers all previous inputs/outputs and adds them to the context that is passed. Let's take a look at using this chain (setting `verbose=True` so we can see the prompt).

```python
from langchain import OpenAI, ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

conversation.predict(input="Hi there!")
```

```pycon
> Entering new chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there!
AI:

> Finished chain.
' Hello! How are you today?'
```

```python
conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
```

```pycon
> Entering new chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there!
AI:  Hello! How are you today?
Human: I'm doing well! Just having a conversation with an AI.
AI:

> Finished chain.
" That's great! What would you like to talk about?"
```