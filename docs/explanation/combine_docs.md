# Context Guided Generation

Language models are trained on large amounts of unstructured data,
which makes the really good at general purpose text generation.
However, when you want the language model to base its reply on some specific
piece of text, it is useful to guide its generation by including those pieces
of text in the context.

There are a few different ways to do this. Note that there is not necessarily
one "best" way to do this - they all have their own pros and cons. We will cover these
methods and the pros/cons in this document. Also note that these methods
are NOT tied to a specific prompt - although we provide basic prompts
for all methods for an easy quick start, you can edit/modify/improve these prompts as you see fit.

The methods