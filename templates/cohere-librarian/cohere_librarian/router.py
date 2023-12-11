from langchain.schema.runnable import RunnableBranch
from langchain.schema.output_parser import StrOutputParser
from .chat import chat
from .blurb_matcher import book_rec_chain
from  .library_info import library_info
from langchain.prompts import ChatPromptTemplate

chain = (
    ChatPromptTemplate.from_template(
        """Given the user message below, classify it as either being about `recommendation` or `other`.

Do not respond with more than one word.


{message}


Classification:"""
    )
    | chat
    | StrOutputParser()
)

extract_op_field = lambda x: x["output_text"]

branch = RunnableBranch(
    (lambda x: "recommendation" in x["topic"].lower(), book_rec_chain | extract_op_field),
    { "message": lambda x: x["message"] } | library_info
)

branched_chain = {"topic": chain, "message": lambda x: x["message"]} | branch