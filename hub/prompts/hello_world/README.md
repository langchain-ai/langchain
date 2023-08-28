# Description of Hello World

Basic prompt designed to be use as a test case, will just instruct the LLM to say "Hello World".


## Inputs

This prompt doesn't have any inputs.


## Usage

Below is a code snippet for how to use the prompt.

```python
from langchain.prompts import load_prompt
from langchain.chains import LLMChain

llm = ...
prompt = load_prompt('lc://prompts/hello-world/<file-name>')
chain = LLMChain(llm=llm, prompt=prompt)
```
