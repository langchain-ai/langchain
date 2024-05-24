# PremAI

[PremAI](https://premai.io/) is an all-in-one platform that simplifies the creation of robust, production-ready applications powered by Generative AI. By streamlining the development process, PremAI allows you to concentrate on enhancing user experience and driving overall growth for your application. You can quickly start using our platform [here](https://docs.premai.io/quick-start).


## ChatPremAI

This example goes over how to use LangChain to interact with different chat models with `ChatPremAI`

### Installation and setup

We start by installing `langchain` and `premai-sdk`. You can type the following command to install:

```bash
pip install premai langchain
```

Before proceeding further, please make sure that you have made an account on PremAI and already created a project. If not, please refer to the [quick start](https://docs.premai.io/introduction) guide to get started with the PremAI platform. Create your first project and grab your API key.

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatPremAI
```

### Setup PremAI client in LangChain

Once we imported our required modules, let's setup our client. For now let's assume that our `project_id` is `8`. But make sure you use your project-id, otherwise it will throw error.

To use langchain with prem, you do not need to pass any model name or set any parameters with our chat-client. By default it will use the model name and parameters used in the [LaunchPad](https://docs.premai.io/get-started/launchpad). 

> Note: If you change the `model` or any other parameters like `temperature`  or `max_tokens` while setting the client, it will override existing default configurations, that was used in LaunchPad.   


```python
import os
import getpass

if "PREMAI_API_KEY" not in os.environ:
    os.environ["PREMAI_API_KEY"] = getpass.getpass("PremAI API Key:")

chat = ChatPremAI(project_id=8)
```

### Chat Completions

`ChatPremAI` supports two methods: `invoke` (which is the same as `generate`) and `stream`. 

The first one will give us a static result. Whereas the second one will stream tokens one by one. Here's how you can generate chat-like completions. 

```python
human_message = HumanMessage(content="Who are you?")

chat.invoke([human_message])
```

You can provide system prompt here like this:

```python
system_message = SystemMessage(content="You are a friendly assistant.")
human_message = HumanMessage(content="Who are you?")

chat.invoke([system_message, human_message])
```

You can also change generation parameters while calling the model. Here's how you can do that:

```python
chat.invoke(
    [system_message, human_message],
    temperature = 0.7, max_tokens = 20, top_p = 0.95
)
```

> If you are going to place system prompt here, then it will override your system prompt that was fixed while deploying the application from the platform. 

> Please note that the current version of ChatPremAI does not support parameters: [n](https://platform.openai.com/docs/api-reference/chat/create#chat-create-n) and [stop](https://platform.openai.com/docs/api-reference/chat/create#chat-create-stop). 

### Streaming

In this section, let's see how we can stream tokens using langchain and PremAI. Here's how you do it. 

```python
import sys

for chunk in chat.stream("hello how are you"):
    sys.stdout.write(chunk.content)
    sys.stdout.flush()
```

Similar to above, if you want to override the system-prompt and the generation parameters, you need to add the following:

```python
import sys

for chunk in chat.stream(
    "hello how are you",
    system_prompt = "You are an helpful assistant", temperature = 0.7, max_tokens = 20
):
    sys.stdout.write(chunk.content)
    sys.stdout.flush()
```

This will stream tokens one after the other.

## PremEmbeddings

In this section we are going to dicuss how we can get access to different embedding model using `PremEmbeddings` with LangChain. Lets start by importing our modules and setting our API Key. 

```python
import os
import getpass
from langchain_community.embeddings import PremEmbeddings


if os.environ.get("PREMAI_API_KEY") is None:
    os.environ["PREMAI_API_KEY"] = getpass.getpass("PremAI API Key:")

```

We support lots of state of the art embedding models. You can view our list of supported LLMs and embedding models [here](https://docs.premai.io/get-started/supported-models). For now let's go for `text-embedding-3-large` model for this example. . 

```python

model = "text-embedding-3-large"
embedder = PremEmbeddings(project_id=8, model=model)

query = "Hello, this is a test query"
query_result = embedder.embed_query(query)

# Let's print the first five elements of the query embedding vector

print(query_result[:5])
```
<Note>
Setting `model_name` argument in mandatory for PremAIEmbeddings unlike chat. 
</Note>

Finally, let's embed some sample document

```python
documents = [
    "This is document1",
    "This is document2",
    "This is document3"
]

doc_result = embedder.embed_documents(documents)

# Similar to the previous result, let's print the first five element
# of the first document vector

print(doc_result[0][:5])
```

```python
print(f"Dimension of embeddings: {len(query_result)}")
```
Dimension of embeddings: 3072

```python
doc_result[:5]
```
>Result:
>
>[-0.02129288576543331,
 0.0008162345038726926,
 -0.004556538071483374,
 0.02918623760342598,
 -0.02547479420900345]