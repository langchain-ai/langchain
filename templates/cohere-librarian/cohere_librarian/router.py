from langchain.schema.runnable import RunnableBranch
from langchain.schema.output_parser import StrOutputParser
from .chat import chat
from .blurb_matcher import book_rec_chain
from  .library_info import library_info
from langchain.prompts import ChatPromptTemplate
from .rag import librarian_rag

chain = (
    ChatPromptTemplate.from_template(
        """Given the user message below, classify it as either being about `recommendation`, `library` or `other`.

'{message}'

Respond with just one word. For example, if the message is about a book recommendation, respond with `recommendation`.
"""
    )
    | chat
    | StrOutputParser()
)

extract_op_field = lambda x: x["output_text"]

branch = RunnableBranch(
    (lambda x: "recommendation" in x["topic"].lower(), book_rec_chain | extract_op_field),
    (lambda x: "library" in x["topic"].lower(), { "message": lambda x: x["message"] } | library_info),
    librarian_rag
)

branched_chain = {"topic": chain, "message": lambda x: x["message"]} | branch