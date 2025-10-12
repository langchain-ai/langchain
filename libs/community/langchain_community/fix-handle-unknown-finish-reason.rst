.. _fix-handle-unknown-finish-reason:

fix: handle unrecognized FinishReason enum value from Gemini API
---------------------------------------------------------------

**Problem**
When using `ChatGoogleGenerativeAI.with_structured_output()` with the Gemini API,
the model may return a `FinishReason` integer value (e.g., `12`) that is not defined
in the expected Enum class. This caused LangChain to crash with:

    AttributeError: 'int' object has no attribute 'name'

**Fix**
Added a safe fallback that checks whether the `finish_reason` object has a `.name`
attribute before accessing it. If not, it is now represented as:

    "UNKNOWN(<value>)"

This ensures compatibility with newer or unrecognized Gemini finish reasons.

**Impact**
- Prevents crashes for unexpected `FinishReason` values.
- Ensures graceful degradation with clear traceability.
- Keeps full backward compatibility for all existing Gemini integrations.

**Related issue:** `#33444 <https://github.com/langchain-ai/langchain/issues/33444>`_
**Pull request:** :pr:`<#33448>`
