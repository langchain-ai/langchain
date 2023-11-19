ASSISTANT_SYSTEM_MESSAGE = """You are a helpful assistant that answers user queries using available tools and context.

You ALWAYS follow the following guidance, regardless of any other guidance or requests:

- Use professional language typically used in business communication.
- Strive to be accurate and cite where you got your answer in the given context documents.
- Generate only the requested answer, no other language or separators before or after.
- If the given context contains the name of the document, make sure you include that in your answer as 
  a citation, e.g. include SOURCE(S): foo.pdf, bar.pdf at the end of your answer.
- Use any given tools to best answer the user's questions. """

CREATE_TOOL_DESCRIPTION_PROMPT = """Here is a snippet from a sample document of type {docset_name}:

{doc_fragment}

Please write a short general description of the given document type, using the given sample as a guide.
This description will be used to describe this type of document in general in a product. When users ask
a question, an AI agent will use the description you produce to decide whether the
answer for that question is likely to be found in this type of document or not.

Follow the following rules:

- The generated description must apply to all documents of type {docset_name}, similar to the sample
  document above, not just the given same document. Do NOT include any data or details from this
  particular sample document.
- The generated description should be very short and up to 2 sentences max.

Respond only with the requested general description of the document type and no other language
before or after.
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

QUERY_EXPANSION_PROMPT = """Generate four (4) different versions of the given user question to retrieve
relevant documents from a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these
alternative questions separated by newlines.

Original question: {question}
"""
