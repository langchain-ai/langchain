import pydantic

# patch validation for Pydantic 2.7 (issue due to OpenAI SDK)
try:
    import pydantic._internal._model_construction

    # Patch both validation functions to avoid OpenAI SDK compatibility issues
    pydantic._internal._model_construction.is_valid_field_name = lambda name: True  # noqa: ARG005
    pydantic._internal._model_construction.is_valid_privateattr_name = lambda name: True  # noqa: ARG005
except (AttributeError, ImportError):
    pass
