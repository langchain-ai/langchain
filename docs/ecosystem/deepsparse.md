# DeepSparse

This page covers how to use the `deepsparse` library within LangChain.
DeepSparse is an inference runtime offering GPU-class performance on CPUs and APIs to integrate ML into your application.

## Installation
- Install the Python package with `pip install deepsparse`

### DeepSparse

To use DeepSparse, you need to provide the path of the zoo stub and then call the model:

```python
from langchain.transformers.neuralmagic import DeepSparse

meta_agent = DeepSparse(model="zoo:nlp/<PLACEHOLDER>")
predict = meta_agent("i'm a prompt")
```