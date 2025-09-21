import pydantic

# patch validation for Pydantic 2.7 (issue due to OpenAI SDK)
pydantic._internal._model_construction.is_valid_field_name = lambda name: True
