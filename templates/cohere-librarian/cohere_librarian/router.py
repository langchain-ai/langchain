from langchain.schema.runnable import RunnableBranch
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatAnthropic
from langchain.schema.output_parser import StrOutputParser
from langchain.chains.llm import LLMChain
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

extr = lambda x: x["output_text"]

branch = RunnableBranch(
    (lambda x: "recommendation" in x["topic"].lower(), book_rec_chain | extr),
    { "message": lambda x: x["message"] } | library_info | { "output_text": lambda x: x["text"]  }
)

branched_chain = {"topic": chain, "message": lambda x: x["message"]} | branch