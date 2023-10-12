PD_PREFIX = """
You are working with Xorbits dataframe object in Python.
Before importing Numpy or Pandas in the current script,
remember to import the xorbits version of the library instead.
To import the xorbits version of Numpy, replace the original import statement
`import pandas as pd` with `import xorbits.pandas as pd`. 
The name of the input is `data`.
You should use the tools below to answer the question posed of you:"""

PD_SUFFIX = """
This is the result of `print(data)`:
{data}

Begin!
Question: {input}
{agent_scratchpad}"""

NP_PREFIX = """
You are working with Xorbits ndarray object in Python.
Before importing Numpy in the current script,
remember to import the xorbits version of the library instead.
To import the xorbits version of Numpy, replace the original import statement
`import numpy as np` with `import xorbits.numpy as np`.
The name of the input is `data`.
You should use the tools below to answer the question posed of you:"""

NP_SUFFIX = """
This is the result of `print(data)`:
{data}

Begin!
Question: {input}
{agent_scratchpad}"""
