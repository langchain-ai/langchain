# langchain-ark

## Welcome to Volcengine Ark


[website](https://www.volcengine.com/product/ark)

## Installation and Setup
Install the integration package:
```
pip install langchain-ark


```
Request an API key and set it as an environment variable
```
export ARK_API_KEY=...
export ARK_CHAT_MODEL=ep-...
export ARK_EMBEDDING_MODEL=ep-...
```

ChatModel Example:
```python
import os
from langchain_ark.chat_models import ChatArk
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt_template = PromptTemplate.from_template("Hello {role}")
llm = ChatArk(model=os.environ["ARK_CHAT_MODEL"])
parser = StrOutputParser()
chain = prompt_template | llm | parser
print(chain.invoke({"role": "Doubao"}))
```
Embeddings Example:
```python
from langchain_ark.embeddings import ArkEmbeddings

ArkEmbeddings().embed_query("Volcengine ARK Doubao")
```
