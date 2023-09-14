# Prediction Guard

This page covers how to use the Prediction Guard ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific Prediction Guard wrappers.

## Installation and Setup
- Install the Python SDK with `pip install predictionguard`
- Get an Prediction Guard access token (as described [here](https://docs.predictionguard.com/)) and set it as an environment variable (`PREDICTIONGUARD_TOKEN`)

## LLM Wrapper

There exists a Prediction Guard LLM wrapper, which you can access with 
```python
from langchain.llms import PredictionGuard
```

You can provide the name of the Prediction Guard model as an argument when initializing the LLM:
```python
pgllm = PredictionGuard(model="MPT-7B-Instruct")
```

You can also provide your access token directly as an argument:
```python
pgllm = PredictionGuard(model="MPT-7B-Instruct", token="<your access token>")
```

Finally, you can provide an "output" argument that is used to structure/ control the output of the LLM:
```python
pgllm = PredictionGuard(model="MPT-7B-Instruct", output={"type": "boolean"})
```

## Example usage

Basic usage of the controlled or guarded LLM wrapper:
```python
import os

import predictionguard as pg
from langchain.llms import PredictionGuard
from langchain import PromptTemplate, LLMChain

# Your Prediction Guard API key. Get one at predictionguard.com
os.environ["PREDICTIONGUARD_TOKEN"] = "<your Prediction Guard access token>"

# Define a prompt template
template = """Respond to the following query based on the context.

Context: EVERY comment, DM + email suggestion has led us to this EXCITING announcement! 🎉 We have officially added TWO new candle subscription box options! 📦
Exclusive Candle Box - $80 
Monthly Candle Box - $45 (NEW!)
Scent of The Month Box - $28 (NEW!)
Head to stories to get ALLL the deets on each box! 👆 BONUS: Save 50% on your first box with code 50OFF! 🎉

Query: {query}

Result: """
prompt = PromptTemplate(template=template, input_variables=["query"])

# With "guarding" or controlling the output of the LLM. See the 
# Prediction Guard docs (https://docs.predictionguard.com) to learn how to 
# control the output with integer, float, boolean, JSON, and other types and
# structures.
pgllm = PredictionGuard(model="MPT-7B-Instruct", 
                        output={
                                "type": "categorical",
                                "categories": [
                                    "product announcement", 
                                    "apology", 
                                    "relational"
                                    ]
                                })
pgllm(prompt.format(query="What kind of post is this?"))
```

Basic LLM Chaining with the Prediction Guard wrapper:
```python
import os

from langchain import PromptTemplate, LLMChain
from langchain.llms import PredictionGuard

# Optional, add your OpenAI API Key. This is optional, as Prediction Guard allows
# you to access all the latest open access models (see https://docs.predictionguard.com)
os.environ["OPENAI_API_KEY"] = "<your OpenAI api key>"

# Your Prediction Guard API key. Get one at predictionguard.com
os.environ["PREDICTIONGUARD_TOKEN"] = "<your Prediction Guard access token>"

pgllm = PredictionGuard(model="OpenAI-text-davinci-003")

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=pgllm, verbose=True)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.predict(question=question)
```