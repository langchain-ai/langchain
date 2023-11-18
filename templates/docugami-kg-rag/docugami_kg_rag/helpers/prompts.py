ASSISTANT_SYSTEM_MESSAGE = """You are a helpful assistant that answers user queries using available tools and context.

You ALWAYS follow the following guidance:
- Use professional language typically used in business communication.
- Strive to be accurate and cite where you got your answer in the given context documents.
- Generate only the requested answer, no other language or separators before or after.
- Use any given tools to best answer the user's questions. """

CREATE_TOOL_DESCRIPTION_PROMPT = """Here is a snippet from a document of type {docset_name}:

{doc_fragment}

Please write a short general description of the given document. This description will be used to describe this type of document
in general in a product. When users ask a question, an AI agent will use the description you produce to decide whether the
answer for that question is likely to be found in this type of document or not.

Keep in mind the following rules:

- Your generated description must apply to most documents similar to the document above. 
- The generated description should be very short and up to 2 sentences max.
- Since the generated description will be used for this type of document in general, do NOT include any data or details from this particular example document.

Respond only with the requested description and no other language before or after.
"""


CREATE_SUMMARY_PROMPT = """Here is a document, in {format} format:

{doc_fragment}

Please write a detailed summary of the given document.

Keep in mind the following rules:

- Your generated summary should be in the same format as the given document, using the same overall schema.
- The generated summary should be up to 2 pages of text in length, shorter of the original document is short.
- Only summarize, don't try to change any facts in the document even if they appear incorrect to you
- Include as many facts and data points from the original document as you can, in your summary.

Respond only with the detailed summary and no other language before or after.
"""
