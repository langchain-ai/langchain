"""**HuggingFace Injection Identifier** is a tool that uses
[HuggingFace Prompt Injection model](https://huggingface.co/deepset/deberta-v3-base-injection)
to detect prompt injection attacks.
"""

from langchain_experimental.prompt_injection_identifier.hugging_face_identifier import (
    HuggingFaceInjectionIdentifier,
)

__all__ = ["HuggingFaceInjectionIdentifier"]
