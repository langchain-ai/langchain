from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch

from .blurb_matcher import book_rec_chain
from .chat import chat
from .library_info import library_info
from .rag import librarian_rag

chain = (
    ChatPromptTemplate.from_template(
        """Given the user message below,
classify it as either being about `recommendation`, `library` or `other`.

'{message}'

Respond with just one word.
For example, if the message is about a book recommendation,respond with 
`recommendation`.
"""
    )
    | chat
    | StrOutputParser()
)


def extract_op_field(x):
    return x["output_text"]


branch = RunnableBranch(
    (
        lambda x: "recommendation" in x["topic"].lower(),
        book_rec_chain | extract_op_field,
    ),
    (
        lambda x: "library" in x["topic"].lower(),
        {"message": lambda x: x["message"]} | library_info,
    ),
    librarian_rag,
)

branched_chain = {"topic": chain, "message": lambda x: x["message"]} | branch
