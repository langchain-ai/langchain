# Pydantic compatibility

- Pydantic v2 was released in June, 2023 (https://docs.pydantic.dev/2.0/blog/pydantic-v2-final/)
- v2 contains has a number of breaking changes (https://docs.pydantic.dev/2.0/migration/)
- Pydantic v2 and v1 are under the same package name, so both versions cannot be installed at the same time

## LangChain Pydantic migration plan

As of `langchain>=0.0.267`, LangChain will allow users to install either Pydantic V1 or V2. 
   * Internally LangChain will continue to [use V1](https://docs.pydantic.dev/latest/migration/#continue-using-pydantic-v1-features).
   * During this time, users can pin their pydantic version to v1 to avoid breaking changes, or start a partial
   migration using pydantic v2 throughout their code, but avoiding mixing v1 and v2 code for LangChain (see below).

User can either pin to pydantic v1, and upgrade their code in one go once LangChain has migrated to v2 internally, or they can start a partial migration to v2, but must avoid mixing v1 and v2 code for LangChain.

Below are two examples of showing how to avoid mixing pydantic v1 and v2 code in
the case of inheritance and in the case of passing objects to LangChain.

**Example 1: Extending via inheritance**

**YES** 

```python
from pydantic.v1 import root_validator, validator

class CustomTool(BaseTool): # BaseTool is v1 code
    x: int = Field(default=1)

    def _run(*args, **kwargs):
        return "hello"

    @validator('x') # v1 code
    @classmethod
    def validate_x(cls, x: int) -> int:
        return 1
    

CustomTool(
    name='custom_tool',
    description="hello",
    x=1,
)
```

Mixing Pydantic v2 primitives with Pydantic v1 primitives can raise cryptic errors

**NO** 

```python
from pydantic import Field, field_validator # pydantic v2

class CustomTool(BaseTool): # BaseTool is v1 code
    x: int = Field(default=1)

    def _run(*args, **kwargs):
        return "hello"

    @field_validator('x') # v2 code
    @classmethod
    def validate_x(cls, x: int) -> int:
        return 1
    

CustomTool( 
    name='custom_tool',
    description="hello",
    x=1,
)
```

**Example 2: Passing objects to LangChain**

**YES**

```python
from langchain.tools.base import Tool
from pydantic.v1 import BaseModel, Field # <-- Uses v1 namespace

class CalculatorInput(BaseModel):
    question: str = Field()

Tool.from_function( # <-- tool uses v1 namespace
    func=lambda question: 'hello',
    name="Calculator",
    description="useful for when you need to answer questions about math",
    args_schema=CalculatorInput
)
```

**NO**

```python
from langchain.tools.base import Tool
from pydantic import BaseModel, Field # <-- Uses v2 namespace

class CalculatorInput(BaseModel):
    question: str = Field()

Tool.from_function( # <-- tool uses v1 namespace
    func=lambda question: 'hello',
    name="Calculator",
    description="useful for when you need to answer questions about math",
    args_schema=CalculatorInput
)
```