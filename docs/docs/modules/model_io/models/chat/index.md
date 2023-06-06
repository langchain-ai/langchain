---
sidebar_position: 1
---
# Chat models

Chat models are a variation on language models.
While chat models use language models under the hood, the interface they expose is a bit different.
Rather than expose a "text in, text out" API, they expose an interface where "chat messages" are the inputs and outputs.

Chat model APIs are fairly new, so we are still figuring out the correct abstractions.

The following sections of documentation are provided:

- [How-to guides](./examples/few_shot_examples.html): Walkthroughs of core functionality, like streaming, creating chat prompts, etc.

- [Integrations](./integrations.html): How to use different chat model providers (OpenAI, Hugging Face, etc).

## Getting Started

The chat model interface is based around messages rather than raw text.

```python
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0)
```

You can get chat completions by passing one or more messages to the chat model. The response will be a message. The types of messages currently supported in LangChain are `AIMessage`, `HumanMessage`, `SystemMessage`, and `ChatMessage` -- `ChatMessage` takes in an arbitrary role parameter. Most of the time, you'll just be dealing with `HumanMessage`, `AIMessage`, and `SystemMessage`


```python
chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
```

<CodeOutputBlock lang="python">

```
    AIMessage(content="J'aime programmer.", additional_kwargs={})
```

</CodeOutputBlock>

OpenAI's chat model supports multiple messages as input. See [here](https://platform.openai.com/docs/guides/chat/chat-vs-completions) for more information. Here is an example of sending a system and user message to the chat model:


```python
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="I love programming.")
]
chat(messages)
```

<CodeOutputBlock lang="python">

```
    AIMessage(content="J'aime programmer.", additional_kwargs={})
```

</CodeOutputBlock>

You can go one step further and generate completions for multiple sets of messages using `generate`. This returns an `LLMResult` with an additional `message` parameter.


```python
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
```

<CodeOutputBlock lang="python">

```
    LLMResult(generations=[[ChatGeneration(text="J'aime programmer.", generation_info=None, message=AIMessage(content="J'aime programmer.", additional_kwargs={}))], [ChatGeneration(text="J'aime l'intelligence artificielle.", generation_info=None, message=AIMessage(content="J'aime l'intelligence artificielle.", additional_kwargs={}))]], llm_output={'token_usage': {'prompt_tokens': 57, 'completion_tokens': 20, 'total_tokens': 77}})
```

</CodeOutputBlock>

You can recover things like token usage from this LLMResult


```python
result.llm_output
```

<CodeOutputBlock lang="python">

```
    {'token_usage': {'prompt_tokens': 57,
      'completion_tokens': 20,
      'total_tokens': 77}}
```

</CodeOutputBlock>

## PromptTemplates

You can make use of templating by using a `MessagePromptTemplate`. You can build a `ChatPromptTemplate` from one or more `MessagePromptTemplates`. You can use `ChatPromptTemplate`'s `format_prompt` -- this returns a `PromptValue`, which you can convert to a string or Message object, depending on whether you want to use the formatted value as input to an llm or chat model.

For convenience, there is a `from_template` method exposed on the template. If you were to use this template, this is what it would look like:


```python
template="You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
```


```python
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# get a chat completion from the formatted messages
chat(chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_messages())
```

<CodeOutputBlock lang="python">

```
    AIMessage(content="J'adore la programmation.", additional_kwargs={})
```

</CodeOutputBlock>

If you wanted to construct the MessagePromptTemplate more directly, you could create a PromptTemplate outside and then pass it in, eg:


```python
prompt=PromptTemplate(
    template="You are a helpful assistant that translates {input_language} to {output_language}.",
    input_variables=["input_language", "output_language"],
)
system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
```

## LLMChain
You can use the existing LLMChain in a very similar way to before - provide a prompt and a model.


```python
chain = LLMChain(llm=chat, prompt=chat_prompt)
```


```python
chain.run(input_language="English", output_language="French", text="I love programming.")
```

<CodeOutputBlock lang="python">

```
    "J'adore la programmation."
```

</CodeOutputBlock>

## Streaming

Streaming is supported for `ChatOpenAI` through callback handling.


```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
resp = chat([HumanMessage(content="Write me a song about sparkling water.")])
```

<CodeOutputBlock lang="python">

```
    
    
    Verse 1:
    Bubbles rising to the top
    A refreshing drink that never stops
    Clear and crisp, it's pure delight
    A taste that's sure to excite
    
    Chorus:
    Sparkling water, oh so fine
    A drink that's always on my mind
    With every sip, I feel alive
    Sparkling water, you're my vibe
    
    Verse 2:
    No sugar, no calories, just pure bliss
    A drink that's hard to resist
    It's the perfect way to quench my thirst
    A drink that always comes first
    
    Chorus:
    Sparkling water, oh so fine
    A drink that's always on my mind
    With every sip, I feel alive
    Sparkling water, you're my vibe
    
    Bridge:
    From the mountains to the sea
    Sparkling water, you're the key
    To a healthy life, a happy soul
    A drink that makes me feel whole
    
    Chorus:
    Sparkling water, oh so fine
    A drink that's always on my mind
    With every sip, I feel alive
    Sparkling water, you're my vibe
    
    Outro:
    Sparkling water, you're the one
    A drink that's always so much fun
    I'll never let you go, my friend
    Sparkling
```

</CodeOutputBlock>
