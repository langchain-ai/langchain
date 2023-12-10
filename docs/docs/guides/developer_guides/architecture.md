# Architecture

The following packages and/or folders exist inside the repo:

- `langchain-core`: Base abstractions and LangChain Expression Language.
- `langchain`: Chains, agents, and retrieval methods that make up the cognitive architecture of your application.
- `langchain-community`: Community maintained integrations.
- `langchain-{integration_partner}`: Dedicated packages for specific integrations.
- `langchain-experimental`: More experimental or unsafe chains, agents, and retrieval methods.
- LangChain Templates: Chains, agents, and retrieval methods exposed as a template in a ready-to-deploy format.

The following packages or services exist outside of the repo but contribute to the LangChain experience

- `langserve`: A Python library for easy deployment of your LangChain application
- LangSmith: A platform to help with debugging, logging, testing, monitoring, and data labeling for your LLM application.
- LangSmith Prompt Hub: A platform to manage prompts, including exploring public prompts.
- Hosted LangServe: A platform for easy one-click deployment of your LangServe application (coming soon).

## LangChain Core

`langchain-core` consists of core abstractions and a run time to join those components together.

### Simple and Modular Abstractions

The base abstractions of LangChain are designed to be as modular and simple as possible. Examples of these abstractions include those for language models, document loaders, embedding models, vectorstores, retrievers, and more. The benefit of having these abstractions is that any provider can implement the required interface and then easily be used in the rest of LangChain.

These are NOT high level or end-to-end abstractions. They simply represent common interfaces for the necessary components. For example: LLMs are just text in, text out. Retrievers are text in, documents out. The low-level and flexible abstractions have made LangChain become the standard for how providers and partners expose their services to the whole GenAI ecosystem, leading to over 700 different integrations to date.

Many frameworks AND applications are either build on top or are interoperable with these abstractions. This includes frameworks like [funcchain](https://github.com/shroominic/funcchain), [langchain-decorators](https://github.com/ju-bezdek/langchain-decorators), [gpt-researcher](https://github.com/assafelovic/gpt-researcher), [gpt-engineer](https://github.com/AntonOsika/gpt-engineer), [llama-index](https://github.com/run-llama/llama_index), [pandas-ai](https://github.com/gventuri/pandas-ai), and [CrewAI](https://github.com/joaomdmoura/CrewAI). Having a LangChain integration is the most effective way to make sure your tool is compatible with a large part of the ecosystem. In addition, over 30k applications are built on top of LangChain. Some of these need to implement custom components. By making our abstractions simple and modular we have made this easy and painless to do.

### LangChain Expression Language

[LangChain Expression Language](https://python.langchain.com/docs/expression_language/) allows users to compose arbitrary sequences together and get several benefits that are important when building LLM applications. We call these sequences “runnables”.

All runnables expose the same interface with single, batch, streaming and async methods. This is useful because when building an LLM application it is not enough to have a single sync interface. Batch is needed for efficient processing of many inputs. Streaming (and streaming of intermediate steps) is needed to show the user that progress is being made. Async interfaces are nice when moving into production. Rather than having to write multiple implementations for all of those, LangChain Expression Language allows you to write a runnable once and invoke it in many different ways.

We have also written runnables to do orchestration of common (but annoying) tasks. All runnables expose a `.map` method, which applies that runnable to all elements of a list in parallel. All runnables also expose a  `.with_fallbacks` method that allows you to define fallbacks in case that runnables errors. These are orchestration tasks that are common and helpful when building LLM applications.

These runnables are also designed to be inspectable and customizable. Runnables are defined in a declarative way, which makes it much easier to understand the logic and modify parts.

Finally, runnables have best-in-class observability through a seamless integration with [LangSmith](https://smith.langchain.com/). LLM applications are incredibly tricky to debug. Understanding what the exact sequence of steps is, what exactly the inputs are, and what exactly the outputs are can greatly increase your prompt velocity. LangSmith helps with exactly that.

## LangChain Community

A huge part of LangChain is our partners. We have nearly 700 integrations, from document loaders to LLMs to vectorstores to toolkits. In all cases we have strived to make it as easy to add integrations as possible, and thank the community and our partners for working with us.

`langchain-community` contains all third party integrations, ranging from LLM providers to vectorstores to agent toolkits. All dependencies here are optional. This has the benefit of making `langchain-community` a pretty lightweight package. As a result, all imports of third party SDKs are done in an optional (lazy) way.

## Integration-Specific Packages

(Coming Soon)

The largest and most important integrations are split out into their own packages. For example: `langchain-openai`, `langchain-anthropic`, etc. These packages either live inside the LangChain monorepo (`libs/partners`) or in their own individual repositories (depending on the desire of the integration partner). Separating these from `langchain-community` has several benefits. It lets us version these packages by themselves, and if breaking changes are made they can be reflected by appropriate version bumps. It simplifies the testing process, increasing test coverage for these key integrations. It lets us make third party dependencies required for these packages, making installation easier. For packages that are split out into their own repo, it lets other companies own their integration.

## LangChain

The `langchain` package is made up of chains, agents, advanced retrieval methods, and other generalizable orchestration pieces that make up an application’s cognitive architecture.  Some of these are old legacy Chains - like `ConversationalRetrievalChain` which is one of our most popular chains for doing retrieval augmented generation. More and more though, we will move towards chains constructed with LangChain Expression Language.

We are defaulting to LangChain Expression Language for many of the reasons listed above. Top of mind are ease of creating new chains, transparency of what steps are involved, ease of editing those steps, and easy exposure of streaming, batch, and async interfaces.

Some of these chains and agents will be generic. Others will be use case specific. The use case specific ones will overlap with LangChain Templates (see below for more details on those).

## LangChain Experimental

`langchain-experimental` is a place to put more experimental tools, chains, and agents. “Experimental” can mean several things. Currently most of the implementations in `langchain-experimental` are either (1) newer and more more “out-there” ideas that we want to encourage people to use and play around with but aren’t sure if they are the proper abstractions, or (2) potentially dangerous and introduce CVEs (Python REPL, writing and executing SQL). We separate these tools, chains, and agents to convey their experimental nature to end users.

## LangChain Templates

[LangChain Templates](https://github.com/langchain-ai/langchain/tree/master/templates) is the easiest way to get started building a GenAI application. These are end-to-end applications that can be easily deployed with LangServe (more on that later). These are distinct from chains and agents in the `langchain` library in a few ways.

First, they are not part of a Python library but are rather code that is downloaded and part of your application. This has a huge benefit in that they are easily modifiable. One of the main things that we observed over time is that as users would bring LangChain applications into production they had to do some customization of the chain or agent that they started from. At a minimum they had to edit the prompts used, and often they had to modify the internals of the chain, how data flowed through. By putting this logic (both the prompts and orchestration) as part of your application - rather than as part of a libraries source code - it is MUCH more modifiable.

The second big difference is that these come pre-configured with various integrations. While chains and agents often require you pass in LLMs or VectorStores to make them work, templates come with those in place already. Since different LLMs may require different prompting strategies, and different vectorstores may have different parameters, this makes it possible for us to partner with integration partners to deliver templates that best take advantage their particular tech. We’ve already added over 50 different templates - check them out [here](https://templates.langchain.com/).