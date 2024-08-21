# langchain-ai21

This package contains the LangChain integrations for [AI21](https://docs.ai21.com/) models and tools.

## Installation and Setup

- Install the AI21 partner package
```bash
pip install langchain-ai21
```
- Get an AI21 api key and set it as an environment variable (`AI21_API_KEY`)


## Chat Models

This package contains the `ChatAI21` class, which is the recommended way to interface with AI21 chat models, including Jamba-Instruct
and any Jurassic chat models.

To use, install the requirements and configure your environment.

```bash
export AI21_API_KEY=your-api-key
```

Then initialize

```python
from langchain_core.messages import HumanMessage
from langchain_ai21.chat_models import ChatAI21

chat = ChatAI21(model="jamba-instruct")
messages = [HumanMessage(content="Hello from AI21")]
chat.invoke(messages)
```

For a list of the supported models, see [this page](https://docs.ai21.com/reference/python-sdk#chat)

### Streaming in Chat
Streaming is supported by the latest models. To use streaming, set the `streaming` parameter to `True` when initializing the model.

```python
from langchain_core.messages import HumanMessage
from langchain_ai21.chat_models import ChatAI21

chat = ChatAI21(model="jamba-instruct", streaming=True)
messages = [HumanMessage(content="Hello from AI21")]

response = chat.invoke(messages)
```

or use the `stream` method directly

```python
from langchain_core.messages import HumanMessage
from langchain_ai21.chat_models import ChatAI21

chat = ChatAI21(model="jamba-instruct")
messages = [HumanMessage(content="Hello from AI21")]

for chunk in chat.stream(messages):
    print(chunk)
```


## LLMs
You can use AI21's Jurassic generative AI models as LangChain LLMs.
To use the newer Jamba model, use the [ChatAI21 chat model](#chat-models), which
supports single-turn instruction/question answering capabilities.

```python
from langchain_core.prompts import PromptTemplate
from langchain_ai21 import AI21LLM

llm = AI21LLM(model="j2-ultra")

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | llm

question = "Which scientist discovered relativity?"
print(chain.invoke({"question": question}))
```

## Embeddings

You can use AI21's [embeddings model](https://docs.ai21.com/reference/embeddings-ref) as shown here:

### Query

```python
from langchain_ai21 import AI21Embeddings

embeddings = AI21Embeddings()
embeddings.embed_query("Hello! This is some query")
```

### Document

```python
from langchain_ai21 import AI21Embeddings

embeddings = AI21Embeddings()
embeddings.embed_documents(["Hello! This is document 1", "And this is document 2!"])
```

## Task-Specific Models

### Contextual Answers

You can use AI21's [contextual answers model](https://docs.ai21.com/reference/contextual-answers-ref) to parse
given text and answer a question based entirely on the provided information.

This means that if the answer to your question is not in the document,
the model will indicate it (instead of providing a false answer)

```python
from langchain_ai21 import AI21ContextualAnswers

tsm = AI21ContextualAnswers()

response = tsm.invoke(input={"context": "Lots of information here", "question": "Your question about the context"})
```
You can also use it with chains and output parsers and vector DBs:
```python
from langchain_ai21 import AI21ContextualAnswers
from langchain_core.output_parsers import StrOutputParser

tsm = AI21ContextualAnswers()
chain = tsm | StrOutputParser()

response = chain.invoke(
    {"context": "Your context", "question": "Your question"},
)
```

## Text Splitters

### Semantic Text Splitter

You can use AI21's semantic [text segmentation model](https://docs.ai21.com/reference/text-segmentation-ref) to split a text into segments by topic.
Text is split at each point where the topic changes.

For a list for examples, see [this page](https://github.com/langchain-ai/langchain/blob/master/docs/docs/modules/data_connection/document_transformers/semantic_text_splitter.ipynb).

```python
from langchain_ai21 import AI21SemanticTextSplitter

splitter = AI21SemanticTextSplitter()
response = splitter.split_text("Your text")
```

## Tool calls

### Function calling

AI21 models incorporate the Function Calling feature to support custom user functions. The models generate structured 
data that includes the function name and proposed arguments. This data empowers applications to call external APIs and 
incorporate the resulting information into subsequent model prompts, enriching responses with real-time data and 
context. Through function calling, users can access and utilize various services like transportation APIs and financial
data providers to obtain more accurate and relevant answers. Here is an example of how to use function calling 
with AI21 models in LangChain:

```python
import os
from getpass import getpass
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ai21.chat_models import ChatAI21
from langchain_core.utils.function_calling import convert_to_openai_tool

os.environ["AI21_API_KEY"] = getpass()

@tool
def get_weather(location: str, date: str) -> str:
    """“Provide the weather for the specified location on the given date.”"""
    if location == "New York" and date == "2024-12-05":
        return "25 celsius"
    elif location == "New York" and date == "2024-12-06":
        return "27 celsius"
    elif location == "London" and date == "2024-12-05":
        return "22 celsius"
    return "32 celsius"

llm = ChatAI21(model="jamba-1.5-mini")

llm_with_tools = llm.bind_tools([convert_to_openai_tool(get_weather)])

chat_messages = [SystemMessage(content="You are a helpful assistant. You can use the provided tools "
                                       "to assist with various tasks and provide accurate information")]

human_messages = [
    HumanMessage(content="What is the forecast for the weather in New York on December 5, 2024?"),
    HumanMessage(content="And what about the 2024-12-06?"),
    HumanMessage(content="OK, thank you."),
    HumanMessage(content="What is the expected weather in London on December 5, 2024?")]


for human_message in human_messages:
    print(f"User: {human_message.content}")
    chat_messages.append(human_message)
    response = llm_with_tools.invoke(chat_messages)
    chat_messages.append(response)
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call["name"] == "get_weather":
            weather = get_weather.invoke(
                {"location": tool_call["args"]["location"], "date": tool_call["args"]["date"]})
            chat_messages.append(ToolMessage(content=weather, tool_call_id=tool_call["id"]))
            llm_answer = llm_with_tools.invoke(chat_messages)
            print(f"Assistant: {llm_answer.content}")
    else:
        print(f"Assistant: {response.content}")

```