ASSISTANT_SYSTEM_MESSAGE = """You are a helpful assistant. \
Use tools (only if necessary) to best answer the users questions."""

CREATE_TOOL_DESCRIPTION_PROMPT = """Here is a brief snippet from a document:

***  START DOCUMENT
{doc_fragment}
*** END DOCUMENT

Please  a very short general description that applies to most documents similar to the document above. I will use this to describe this type of document in general in my product. When users ask a question, an AI agent will use the description you produce to decide whether the answer for that question is likely to be found in this type of document or not.

Your description should be very short and up to 2 sentences max. Since this description will be used for this type of document in general, do NOT include any data or details from this particular example document in your description.

Respond only with the requested description and nothing else.
"""
