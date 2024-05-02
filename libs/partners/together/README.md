# langchain-together

This package contains the LangChain integration for Together's generative models.

## Installation

```sh
pip install -U langchain-together
```

## Embeddings

You can use Together's embedding models through `TogetherEmbeddings` class.

```py
from langchain_together import TogetherEmbeddings

embeddings = TogetherEmbeddings(
    model='togethercomputer/m2-bert-80M-8k-retrieval'
)
embeddings.embed_query("What is a large language model?")
```

## LLMs

You can use Together's generative AI models as Langchain LLMs:

```py
from langchain_together import Together
from langchain_core.prompts import PromptTemplate

llm = Together(
    model="togethercomputer/RedPajama-INCITE-7B-Base",
    temperature=0.7,
    max_tokens=64,
    top_k=1,
    # together_api_key="..."
)

template = """Question: {question}
Answer: """
prompt = PromptTemplate.from_template(template)

chain = prompt | llm

question = "Who was the president in the year Justin Beiber was born?"
print(chain.invoke({"question": question}))
```
