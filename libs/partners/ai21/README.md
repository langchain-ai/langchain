# langchain-ai21

This package contains the LangChain integrations for [AI21](https://docs.ai21.com/) through their [AI21](https://pypi.org/project/ai21/) SDK.

## Installation and Setup

- Install the AI21 partner package
```bash
pip install langchain-ai21
```
- Get an AI21 api key and set it as an environment variable (`AI21_API_KEY`)


## Chat Models

This package contains the `ChatAI21` class, which is the recommended way to interface with AI21 Chat models.

To use, install the requirements, and configure your environment.

```bash
export AI21_API_KEY=your-api-key
```

Then initialize

```python
from langchain_core.messages import HumanMessage
from langchain_ai21.chat_models import ChatAI21

chat = ChatAI21(model="j2-ultra")
messages = [HumanMessage(content="Hello from AI21")]
chat.invoke(messages)
```

## LLMs
You can use AI21's generative AI models as Langchain LLMs:

```python
from langchain.prompts import PromptTemplate
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

You can use AI21's embeddings models as:

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
