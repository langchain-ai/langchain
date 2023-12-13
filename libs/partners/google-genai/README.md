# langchain-google

This partner package contains the newer Google LangChain integrations.

## Installation

```python
pip install -U langchain-google
```

## Chat Models

This package contains the `ChatGoogleGenerativeAI` class, which is the recommended way to interface with the Google Gemini series of models.

To use, install the requirements, and configure your environment.

```bash
export GOOGLE_API_KEY=your-api-key
```

Then initialize

```python
from langchain_google import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro")
llm.invoke("Sing a ballad of LangChain.")
```

#### Multimodal inputs

Gemini vision model supports image inputs when providing a single chat message. Example:

```
from langchain_google import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
message = HumanMessage(
    contents = []
    {"type": "text", "text": "What's in this image?"}
    {"type": "image_url": "image_url": "data:image/png;base64,abcd124"}
)
llm.invoke(message)
```

The value of `image_url` can be any of the following:

- A public image URL
- An accessible gcs file (e.g., "gcs://path/to/file.png")
- A local file path
- A base64 encoded image (e.g., `data:image/png;base64,abcd124`)
- A PIL image
