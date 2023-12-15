# langchain-google-genai

This package contains the LangChain integrations for Gemini through their generative-ai SDK.

## Installation

```python
pip install -U langchain-google-genai
```

## Chat Models

This package contains the `ChatGoogleGenerativeAI` class, which is the recommended way to interface with the Google Gemini series of models.

To use, install the requirements, and configure your environment.

```bash
export GOOGLE_API_KEY=your-api-key
```

Then initialize

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro")
llm.invoke("Sing a ballad of LangChain.")
```

#### Multimodal inputs

Gemini vision model supports image inputs when providing a single chat message. Example:

```
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
# example
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What's in this image?",
        },  # You can optionally provide text parts
        {"type": "image_url", "image_url": "https://picsum.photos/seed/picsum/200/300"},
    ]
)
llm.invoke([message])
```

The value of `image_url` can be any of the following:

- A public image URL
- An accessible gcs file (e.g., "gcs://path/to/file.png")
- A local file path
- A base64 encoded image (e.g., `data:image/png;base64,abcd124`)
- A PIL image



## Embeddings

This package also adds support for google's embeddings models.

```
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
embeddings.embed_query("hello, world!")
```