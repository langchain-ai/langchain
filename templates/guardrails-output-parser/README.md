# guardrails-output-parser

Uses [guardrails-ai](https://github.com/guardrails-ai/guardrails) to validate output.

This example protects against profanity, but with Guardrails you can protect against a multitude of things.

If Guardrails does not find any profanity, then the translated output is returned as is.

If Guardrails does find profanity, then an empty string is returned.
