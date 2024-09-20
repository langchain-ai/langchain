---
level: error
---

# Pydantic Usage

Please restrict usage of pydantic to the following list of allowed classes and
functions: BaseModel, SecretStr, ValidatorError, Field, ConfigDict, PrivateAttr, model_validator

```grit
engine marzano(0.1)
language python


`from pydantic import $y` where {
    $y <: ! every contains or  {"model_validator", "BaseModel", "field_validator", "SecretStr", "ValidationError", "Field", "ConfigDict", "PrivateAttr"}
}
```
