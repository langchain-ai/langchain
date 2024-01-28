# langchain-google-vertexai

This package contains the LangChain integrations for Google Cloud generative models.

## Installation

```bash
pip install -U langchain-google-vertexai
```

## Chat Models

`ChatVertexAI` class exposes models such as `gemini-pro` and `chat-bison`.

To use, you should have Google Cloud project with APIs enabled, and configured credentials. Initialize the model as:

```python
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model_name="gemini-pro")
llm.invoke("Sing a ballad of LangChain.")
```

You can use other models, e.g. `chat-bison`:
```python
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model_name="chat-bison", temperature=0.3)
llm.invoke("Sing a ballad of LangChain.")
```

#### Multimodal inputs

Gemini vision model supports image inputs when providing a single chat message. Example:

```python
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model_name="gemini-pro-vision")
# example
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What's in this image?",
        },  # You can optionally provide text parts
        {"type": "image_url", "image_url": {"url": "https://picsum.photos/seed/picsum/200/300"}},
    ]
)
llm.invoke([message])
```

The value of `image_url` can be any of the following:

- A public image URL
- An accessible gcs file (e.g., "gcs://path/to/file.png")
- A local file path
- A base64 encoded image (e.g., `data:image/png;base64,abcd124`)


## Embeddings

You can use Google Cloud's embeddings models as:

```python
from langchain_google_vertexai import VertexAIEmbeddings

embeddings = VertexAIEmbeddings()
embeddings.embed_query("hello, world!")
```

## LLMs
You can use Google Cloud's generative AI models as Langchain LLMs:

```python
from langchain.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | llm

question = "Who was the president in the year Justin Beiber was born?"
print(chain.invoke({"question": question}))
```

You can use Gemini and Palm models, including code-generations ones:
```python
from langchain_google_vertexai import VertexAI

llm = VertexAI(model_name="code-bison", max_output_tokens=1000, temperature=0.3)

question = "Write a python function that checks if a string is a valid email address"

output = llm(question)
```
