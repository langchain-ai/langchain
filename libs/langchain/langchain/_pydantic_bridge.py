import pydantic

PYDANTIC_V2 = pydantic.VERSION.startswith("2.")

if PYDANTIC_V2:
    from pydantic import model_validator, ConfigDict
    from pydantic import BaseModel as BM

    class BaseModel(BM):
        model_config = ConfigDict(arbitrary_types_allowed=True)
else:
    from pydantic import root_validator as old_root_validator
    from pydantic import BaseModel as BM

    class BaseModel(BM):

        class Config:
            """Configuration for this pydantic object."""

            arbitrary_types_allowed = True

def root_validator(*args, **kwargs):
    if PYDANTIC_V2:
        decorator = model_validator
    else:
        decorator = old_root_validator

    # Check if it's being called as @root_validator without ()
    if args and callable(args[0]):
        return decorator(args[0])

    # Otherwise, it's being called with arguments as @root_validator(...)
    return decorator