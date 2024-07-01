"""**Data anonymizer** contains both Anonymizers and Deanonymizers.
It uses the [Microsoft Presidio](https://microsoft.github.io/presidio/) library.

**Anonymizers** are used to replace a `Personally Identifiable Information (PII)`
entity text with some other
value by applying a certain operator (e.g. replace, mask, redact, encrypt).

**Deanonymizers** are used to revert the anonymization operation
(e.g. to decrypt an encrypted text).
"""

from langchain_experimental.data_anonymizer.presidio import (
    PresidioAnonymizer,
    PresidioReversibleAnonymizer,
)

__all__ = ["PresidioAnonymizer", "PresidioReversibleAnonymizer"]
