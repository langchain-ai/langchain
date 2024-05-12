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

## Task Specific Models

### Contextual Answers

You can use AI21's contextual answers model to receives text or document, serving as a context,
and a question and returns an answer based entirely on this context.

This means that if the answer to your question is not in the document,
the model will indicate it (instead of providing a false answer)

```python
from langchain_ai21 import AI21ContextualAnswers

tsm = AI21ContextualAnswers()

response = tsm.invoke(input={"context": "Your context", "question": "Your question"})
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

You can use AI21's semantic text splitter to split a text into segments.
Instead of merely using punctuation and newlines to divide the text, it identifies distinct topics that will work well together and will form a coherent piece of text.

For a list for examples, see [this page](https://github.com/langchain-ai/langchain/blob/master/docs/docs/modules/data_connection/document_transformers/semantic_text_splitter.ipynb).

```python
from langchain_ai21 import AI21SemanticTextSplitter

splitter = AI21SemanticTextSplitter()
response = splitter.split_text("Your text")
```