# How to use LangChain with different Pydantic versions

- Pydantic v2 was released in June, 2023 (https://docs.pydantic.dev/2.0/blog/pydantic-v2-final/).
- v2 contains has a number of breaking changes (https://docs.pydantic.dev/2.0/migration/).
- Pydantic 1 End of Life was in June 2024. LangChain will be dropping support for Pydantic 1 in the near future,
and likely migrating internally to Pydantic 2. The timeline is tentatively September. This change will be accompanied by a minor version bump in the main langchain packages to version 0.3.x.

As of `langchain>=0.0.267`, LangChain allows users to install either Pydantic V1 or V2.

Internally, LangChain continues to use the [Pydantic V1](https://docs.pydantic.dev/latest/migration/#continue-using-pydantic-v1-features) via
the v1 namespace of Pydantic 2.

Because Pydantic does not support mixing .v1 and .v2 objects, users should be aware of a number of issues
when using LangChain with Pydantic.

:::caution
While LangChain supports Pydantic V2 objects in some APIs (listed below), it's suggested that users keep using Pydantic V1 objects until LangChain 0.3 is released.
:::


## 1. Passing Pydantic objects to LangChain APIs

Most LangChain APIs for *tool usage* (see list below) have been updated to accept either Pydantic v1 or v2 objects.

* Pydantic v1 objects correspond to subclasses of `pydantic.BaseModel` if `pydantic 1` is installed or subclasses of `pydantic.v1.BaseModel` if `pydantic 2` is installed.
* Pydantic v2 objects correspond to subclasses of `pydantic.BaseModel` if `pydantic 2` is installed.


| API                                    | Pydantic 1 | Pydantic 2                                                     |
|----------------------------------------|------------|----------------------------------------------------------------|
| `BaseChatModel.bind_tools`             | Yes        | langchain-core>=0.2.23, appropriate version of partner package |
| `BaseChatModel.with_structured_output` | Yes        | langchain-core>=0.2.23, appropriate version of partner package |
| `Tool.from_function`                   | Yes        | langchain-core>=0.2.23                                         |
| `StructuredTool.from_function`         | Yes        | langchain-core>=0.2.23                                         |


Partner packages that accept pydantic v2 objects via `bind_tools` or `with_structured_output` APIs:

| Package Name        | pydantic v1 | pydantic v2 |
|---------------------|-------------|-------------|
| langchain-mistralai | Yes         | >=0.1.11    |
| langchain-anthropic | Yes         | >=0.1.21    |
| langchain-robocorp  | Yes         | >=0.0.10    |
| langchain-openai    | Yes         | >=0.1.19    |
| langchain-fireworks | Yes         | >=0.1.5     |
| langchain-aws       | Yes         | >=0.1.15    |

Additional partner packages will be updated to accept Pydantic v2 objects in the future.

If you are still seeing issues with these APIs or other APIs that accept Pydantic objects, please open an issue, and we'll
address it.

Example:

Prior to `langchain-core<0.2.23`, use Pydantic v1 objects when passing to LangChain APIs.


```python
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel # <-- Note v1 namespace

class Person(BaseModel):
    """Personal information"""
    name: str
    
model = ChatOpenAI()
model = model.with_structured_output(Person)

model.invoke('Bob is a person.')
```

After `langchain-core>=0.2.23`, use either Pydantic v1 or v2 objects when passing to LangChain APIs.

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class Person(BaseModel):
    """Personal information"""
    name: str
    
    
model = ChatOpenAI()
model = model.with_structured_output(Person)

model.invoke('Bob is a person.')
```

## 2. Sub-classing LangChain models

Because LangChain internally uses Pydantic v1, if you are sub-classing LangChain models, you should use Pydantic v1
primitives.


**Example 1: Extending via inheritance**

**YES** 

```python
from pydantic.v1 import validator
from langchain_core.tools import BaseTool

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
from langchain_core.tools import BaseTool

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


## 3. Disable run-time validation for LangChain objects used inside Pydantic v2 models

e.g.,

```python
from typing import Annotated

from langchain_openai import ChatOpenAI # <-- ChatOpenAI uses pydantic v1
from pydantic import BaseModel, SkipValidation


class Foo(BaseModel): # <-- BaseModel is from Pydantic v2
    model: Annotated[ChatOpenAI, SkipValidation()]

Foo(model=ChatOpenAI(api_key="hello"))
```

## 4: LangServe cannot generate OpenAPI docs if running Pydantic 2

If you are using Pydantic 2, you will not be able to generate OpenAPI docs using LangServe.

If you need OpenAPI docs, your options are to either install Pydantic 1:

`pip install pydantic==1.10.17`

or else to use the `APIHandler` object in LangChain to manually create the
routes for your API.

See: https://python.langchain.com/v0.2/docs/langserve/#pydantic
