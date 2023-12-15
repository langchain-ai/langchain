"""
**LangChain NVIDIA AI Foundation Model Playground Integration**

This comprehensive module integrates NVIDIA's state-of-the-art AI Foundation Models, featuring advanced models for conversational AI and semantic embeddings, into the LangChain framework. It provides robust classes for seamless interaction with NVIDIA's AI models, particularly tailored for enriching conversational experiences and enhancing semantic understanding in various applications.

**Features:**

1. **Chat Models (`ChatNVIDIA`):** This class serves as the primary interface for interacting with NVIDIA's Foundation chat models. Users can effortlessly utilize NVIDIA's advanced models like 'Mistral' to engage in rich, context-aware conversations, applicable across diverse domains from customer support to interactive storytelling.

2. **Semantic Embeddings (`NVIDIAEmbeddings`):** The module offers capabilities to generate sophisticated embeddings using NVIDIA's AI models. These embeddings are instrumental for tasks like semantic analysis, text similarity assessments, and contextual understanding, significantly enhancing the depth of NLP applications.

**Installation:**

Install this module easily using pip:

```python
pip install langchain-nvidia-ai-endpoints
```

## Utilizing Chat Models:

After setting up the environment, interact with NVIDIA AI Foundation models:
```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA

ai_chat_model = ChatNVIDIA(model="llama2_13b")
response = ai_chat_model.invoke("Tell me about the LangChain integration.")
```

# Generating Semantic Embeddings:

Use NVIDIA's models for creating embeddings, useful in various NLP tasks:

```python
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

embed_model = NVIDIAEmbeddings(model="nvolveqa_40k")
embedding_output = embed_model.embed_query("Exploring AI capabilities.")
```
"""  # noqa: E501

from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA
from langchain_nvidia_ai_endpoints.embeddings import NVIDIAEmbeddings

__all__ = ["ChatNVIDIA", "NVIDIAEmbeddings"]
