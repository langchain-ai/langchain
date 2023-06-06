======================
Quickstart
======================

In this quickstart tutorial you'll build a few simple language model application with LangChain. Along the way you'll learn about the core building blocks of the framework: Models, Prompts, Chains, Agents, and Memory.

Installation
======================

To get started, install LangChain by running:

.. code-block:: bash

    pip install langchain

or if you prefer conda:

.. code-block:: bash

    conda install langchain -c conda-forge

Environment Setup
======================

Using LangChain will usually require integrations with one or more model providers, data stores, APIs, etc.

For this example, we will be using OpenAI's model APIs. First we'll need to install their Python package:

.. code-block:: bash

    pip install openai


Accessing the API requires an API key, which you can get by creating an account and heading [here](https://platform.openai.com/account/api-keys). Once we have a key we'll want to set it as an environment variable by running:

.. code-block:: bash

    export OPENAI_API_KEY="..."


Alternatively, you could do this from inside your Python script or notebook by adding:

.. code-block:: python

    import os

    os.environ["OPENAI_API_KEY"] = "..."


If you'd prefer not to set an environment variable you can pass the key in directly via the `openai_api_key` named parameter when initiating the OpenAI LLM class:

.. code-block:: python

    from langchain.llms import OpenAI

    llm = OpenAI(openai_api_key="...")


Building a Language Model Application
======================

Now we can start building our language model application. LangChain provides many modules that can be used to build language model applications. Modules can be used as standalones in simple applications, or they can be combined to create more complex functionality.

LLMs
=====================
Get predictions from a language model
---------------------

The basic building block of LangChain is the LLM, which takes in text and generates more text.

As an example, suppose we're building an application that generates a company name based on a company description. In order to do this, we first need to import the OpenAI LLM wrapper.

.. code-block:: python

    from langchain.llms import OpenAI


Now we can initialize the wrapper with relevant parameters. In this case, since we want the outputs to be MORE random, we'll initialize our model with a HIGH temperature.

.. code-block:: python

    llm = OpenAI(temperature=0.9)


And now we can pass in text and get predictions!

.. code-block:: python

    llm("What would be a good company name for a company that makes colorful socks?")


.. code-block:: pycon

    Feetful of Fun


For more details on how to use LLMs within LangChain, see the [LLM getting started guide](../modules/models/llms/getting_started.ipynb).


Chat Models
=====================
Get message completions from a chat model
---------------------

Chat models are a variation on language models. While chat models use language models under the hood, the interface they expose is a bit different: rather than expose a "text in, text out" API, they expose an interface where "chat messages" are the inputs and outputs.

Chat model APIs are fairly new, so we are still figuring out the correct abstractions.

You can get chat completions by passing one or more messages to the chat model. The response will be a message. The types of messages currently supported in LangChain are `AIMessage`, `HumanMessage`, `SystemMessage`, and `ChatMessage` -- `ChatMessage` takes in an arbitrary role parameter. Most of the time, you'll just be dealing with `HumanMessage`, `AIMessage`, and `SystemMessage`.

.. code-block:: python

    from langchain.chat_models import ChatOpenAI
    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )

    chat = ChatOpenAI(temperature=0)

You can get completions by passing in a single message.

.. code-block:: python

    chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
    # -> AIMessage(content="J'aime programmer.", additional_kwargs={})


You can also pass in multiple messages for OpenAI's gpt-3.5-turbo and gpt-4 models.

.. code-block:: python

    messages = [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love programming.")
    ]
    chat(messages)
    # -> AIMessage(content="J'aime programmer.", additional_kwargs={})


You can go one step further and generate completions for multiple sets of messages using `generate`. This returns an `LLMResult` with an additional `message` parameter:

.. code-block:: python

    batch_messages = [
        [
            SystemMessage(content="You are a helpful assistant that translates English to French."),
            HumanMessage(content="I love programming.")
        ],
        [
            SystemMessage(content="You are a helpful assistant that translates English to French."),
            HumanMessage(content="I love artificial intelligence.")
        ],
    ]
    result = chat.generate(batch_messages)
    result
    # -> LLMResult(generations=[[ChatGeneration(text="J'aime programmer.", generation_info=None, message=AIMessage(content="J'aime programmer.", additional_kwargs={}))], [ChatGeneration(text="J'aime l'intelligence artificielle.", generation_info=None, message=AIMessage(content="J'aime l'intelligence artificielle.", additional_kwargs={}))]], llm_output={'token_usage': {'prompt_tokens': 57, 'completion_tokens': 20, 'total_tokens': 77}})


You can recover things like token usage from this LLMResult:

.. code-block:: python

    result.llm_output['token_usage']
    # -> {'prompt_tokens': 57, 'completion_tokens': 20, 'total_tokens': 77}

Prompt Templates
=====================
Manage model prompts
---------------------

.. tabs::

    .. group-tab:: LLMs

       Most LLM applications do not pass user input directly into to an LLM. Usually they will add the user input to a larger piece of text, called a prompt, that provides additional context on the specific task at hand.

       In the previous example, the text we passed to the model contained instructions to generate a company name. For our application, it'd be great if the user only had to provide the description of a company/product, without having to worry about giving the model instructions.

       With PromptTemplates this is easy! In this case our template would be very simple:

       .. code-block:: python

            from langchain.prompts import PromptTemplate

            prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")

       Now can call the `.format` method with arguments corresponding to our string template to construct the full model input:

       .. code-block:: python

           prompt.format(product="colorful socks")

       .. code-block:: pycon

           What is a good name for a company that makes colorful socks?

       For more details, check out the [Prompts getting started guide](../modules/prompts/chat_prompt_template.ipynb).

    .. group-tab:: Chat models

        Similar to LLMs, you can make use of templating by using a `MessagePromptTemplate`. You can build a `ChatPromptTemplate` from one or more `MessagePromptTemplate`s. You can use `ChatPromptTemplate`'s `format_prompt` -- this returns a `PromptValue`, which you can convert to a string or `Message` object, depending on whether you want to use the formatted value as input to an llm or chat model.

        For convenience, there is a `from_template` method exposed on the template. If you were to use this template, this is what it would look like:

        .. code-block:: python

            from langchain.chat_models import ChatOpenAI
            from langchain.prompts.chat import (
                ChatPromptTemplate,
                SystemMessagePromptTemplate,
                HumanMessagePromptTemplate,
            )

            chat = ChatOpenAI(temperature=0)

            template = "You are a helpful assistant that translates {input_language} to {output_language}."
            system_message_prompt = SystemMessagePromptTemplate.from_template(template)
            human_template = "{text}"
            human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

            # get a chat completion from the formatted messages
            chat(chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_messages())

        .. code-block:: pycon

            AIMessage(content="J'aime programmer.", additional_kwargs={})

Chains
=====================
Combine LLMs and prompts in multi-step workflows
---------------------

.. tabs::

    .. group-tab:: LLMs

        Now that we've got a model and a prompt template, we'll want to combine the two. Chains give us a way to link (or chain) together multiple primitives, like models, prompts, and other chains.

        The simplest and most common type of chain is an LLMChain, which passes an input first to a PromptTemplate and then to an LLM. We can construct an LLM chain from our existing model and prompt template:

        .. code-block:: python

            from langchain.chains import LLMChain

            chain = LLMChain(llm=llm, prompt=prompt)


        Using this chain we can now replace

        .. code-block:: python

            llm("What would be a good company name for a company that makes colorful socks?")

        with

        .. code-block:: python

            chain.run("colorful socks")

        and get the same output (model randomness aside)

        .. code-block:: python

            Feetful of Fun


        There we go, our first chain! Understanding how this simple chain works will set you up well for working with more complex chains.

        For more details, check out the [Chain getting started guide](../modules/chains/getting_started.ipynb).

    .. group-tab:: Chat models

        The `LLMChain` discussed in the above section can be used with chat models as well:

        .. code-block:: python

            from langchain.chat_models import ChatOpenAI
            from langchain import LLMChain
            from langchain.prompts.chat import (
                ChatPromptTemplate,
                SystemMessagePromptTemplate,
                HumanMessagePromptTemplate,
            )

            chat = ChatOpenAI(temperature=0)

            template = "You are a helpful assistant that translates {input_language} to {output_language}."
            system_message_prompt = SystemMessagePromptTemplate.from_template(template)
            human_template = "{text}"
            human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

            chain = LLMChain(llm=chat, prompt=chat_prompt)
            chain.run(input_language="English", output_language="French", text="I love programming.")

        .. code-block:: pycon

            "J'aime programmer."


Agents
======================
Dynamically call chains based on user input
----------------------

.. tabs::

    .. group-tab:: LLMs

        Our first chain ran a pre-determined sequence of steps. To handle complex workflows, we need to be able to dynamically choose actions based on inputs.

        Agents do just this: they use an LLM to determine which actions to take and in what order. Agents are given access to tools, and they repeatedly choose a tool, run the tool, and observe the output until they come up with a final answer.

        To load an agent, you need to choose a(n):

        - LLM: The language model powering the agent.
        - Tool(s): A function that performs a specific duty. This can be things like: Google Search, Database lookup, Python REPL, other chains. For a list of predefined tools and their specifications, see [here](../modules/agents/tools/getting_started.md).
        - Agent name: A string that references a support agent class. Because this notebook focuses on the simplest, highest level API, this only covers using the standard supported agents. If you want to implement a custom agent, see the documentation for custom agents (coming soon). For a list of supported agents and their specifications, see [here](../modules/agents/getting_started.ipynb).

        For this example, we'll be using SerpAPI to query a search engine. You'll need to install the SerpAPI Python package:

        .. code-block:: bash

            pip install google-search-results


        And set the SERPAPI_API_KEY environment variable.

        .. code-block:: python

            import os

            os.environ["SERPAPI_API_KEY"] = "..."


        Now we can get started!

        .. code-block:: python

            from langchain.agents import AgentType, initialize_agent, load_tools
            from langchain.llms import OpenAI

            # The language model we're going to use to control the agent.
            llm = OpenAI(temperature=0)

            # The tools we'll give the Agent access to. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
            tools = load_tools(["serpapi", "llm-math"], llm=llm)

            # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
            agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

            # Let's test it out!
            agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")


        Looking at the trace (which is printed because of the `verbose` flag) we can see the sequence of observations and actions the agent took. First it realized it decided to use the search engine to look up the temperature. It then extracted the temperature from the search result and used the math tool to exponentiate.

        .. code-block:: python

            > Entering new AgentExecutor chain...

            Thought: I need to find the temperature first, then use the calculator to raise it to the .023 power.
            Action: Search
            Action Input: "High temperature in SF yesterday"
            Observation: San Francisco Temperature Yesterday. Maximum temperature yesterday: 57 °F (at 1:56 pm) Minimum temperature yesterday: 49 °F (at 1:56 am) Average temperature ...

            Thought: I now have the temperature, so I can use the calculator to raise it to the .023 power.
            Action: Calculator
            Action Input: 57^.023
            Observation: Answer: 1.0974509573251117

            Thought: I now know the final answer
            Final Answer: The high temperature in SF yesterday in Fahrenheit raised to the .023 power is 1.0974509573251117.

            > Finished chain.

        .. code-block:: pycon

            The high temperature in SF yesterday in Fahrenheit raised to the .023 power is 1.0974509573251117.

    .. group-tab:: Chat models

        Agents can also be used with chat models, you can initialize one using `AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION` as the agent type.

        .. code-block:: python

            from langchain.agents import load_tools
            from langchain.agents import initialize_agent
            from langchain.agents import AgentType
            from langchain.chat_models import ChatOpenAI
            from langchain.llms import OpenAI

            # First, let's load the language model we're going to use to control the agent.
            chat = ChatOpenAI(temperature=0)

            # Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
            llm = OpenAI(temperature=0)
            tools = load_tools(["serpapi", "llm-math"], llm=llm)


            # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
            agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

            # Now let's test it out!
            agent.run("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")

        .. code-block:: pycon

            > Entering new AgentExecutor chain...
            Thought: I need to use a search engine to find Olivia Wilde's boyfriend and a calculator to raise his age to the 0.23 power.
            Action:
            {
              "action": "Search",
              "action_input": "Olivia Wilde boyfriend"
            }

            Observation: Sudeikis and Wilde's relationship ended in November 2020. Wilde was publicly served with court documents regarding child custody while she was presenting Don't Worry Darling at CinemaCon 2022. In January 2021, Wilde began dating singer Harry Styles after meeting during the filming of Don't Worry Darling.
            Thought:I need to use a search engine to find Harry Styles' current age.
            Action:
            {
              "action": "Search",
              "action_input": "Harry Styles age"
            }

            Observation: 29 years
            Thought:Now I need to calculate 29 raised to the 0.23 power.
            Action:
            {
              "action": "Calculator",
              "action_input": "29^0.23"
            }

            Observation: Answer: 2.169459462491557

            Thought:I now know the final answer.
            Final Answer: 2.169459462491557

            > Finished chain.
            '2.169459462491557'

Memory
======================
Add state to chains and agents
----------------------

.. tabs::

    .. group-tab:: LLMs

        The chains and agents we've looked at so far have been stateless, but for many applications it's necessary to reference past interactions. This is clearly the case with a chatbot for example, where you want it to understand new messages in the context of past messages.

        The Memory module gives you a way to maintain application state. The base Memory interface is simple: it lets you update state given the latest run inputs and outputs and it lets you modify (or contextualize) the next input using the stored state.

        There are a number of built-in memory systems. The simplest of these are buffers that prepend the last few inputs/outputs to the current input.
        There are also a number of chains with built-in memory. This notebook walks through using one of those chains (the `ConversationChain`) with two different types of memory.

        By default, the `ConversationChain` has a simple type of memory that remembers all previous inputs/outputs and adds as many of them to the prompt as it can. Let's take a look at using this chain (setting `verbose=True` so we can see the prompt).

        .. code-block:: python

            from langchain import OpenAI, ConversationChain

            llm = OpenAI(temperature=0)
            conversation = ConversationChain(llm=llm, verbose=True)

            conversation.run("Hi there!")

        here's what's going on under the hood

        .. code-block:: pycon

            > Entering new chain...
            Prompt after formatting:
            The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

            Current conversation:

            Human: Hi there!
            AI:

            > Finished chain.

        and here's our final output

        .. code-block:: pycon

            'Hello! How are you today?'

        Now if we run the chain again

        .. code-block:: python

            conversation.run("I'm doing well! Just having a conversation with an AI.")

        we'll see that the full prompt that's passed to the model contains the input and output of our first interaction, along with our latest input

        .. code-block:: pycon

            > Entering new chain...
            Prompt after formatting:
            The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

            Current conversation:

            Human: Hi there!
            AI:  Hello! How are you today?
            Human: I'm doing well! Just having a conversation with an AI.
            AI:

            > Finished chain.

        .. code-block:: pycon

            "That's great! What would you like to talk about?"

    .. group-tab:: Chat models

        You can use Memory with chains and agents initialized with chat models. The main difference between this and Memory for LLMs is that rather than trying to condense all previous messages into a string, we can keep them as their own unique memory object.

        .. code-block:: python

            from langchain.prompts import (
                ChatPromptTemplate,
                MessagesPlaceholder,
                SystemMessagePromptTemplate,
                HumanMessagePromptTemplate
            )
            from langchain.chains import ConversationChain
            from langchain.chat_models import ChatOpenAI
            from langchain.memory import ConversationBufferMemory

            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "The following is a friendly conversation between a human and an AI. The AI is talkative and "
                    "provides lots of specific details from its context. If the AI does not know the answer to a "
                    "question, it truthfully says it does not know."
                ),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ])

            llm = ChatOpenAI(temperature=0)
            memory = ConversationBufferMemory(return_messages=True)
            conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

            conversation.predict(input="Hi there!")

        .. code-block:: pycon

            'Hello! How can I assist you today?'

        .. code-block:: python

            conversation.predict(input="I'm doing well! Just having a conversation with an AI.")

        .. code-block:: pycon

            "That sounds like fun! I'm happy to chat with you. Is there anything specific you'd like to talk about?"

            conversation.predict(input="Tell me about yourself.")
            # -> "Sure! I am an AI language model created by OpenAI. I was trained on a large dataset of text from the internet, which allows me to understand and generate human-like language. I can answer questions, provide information, and even have conversations like this one. Is there anything else you'd like to know about me?"

