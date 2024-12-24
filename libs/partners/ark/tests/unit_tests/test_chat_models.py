import pytest
from dotenv import load_dotenv

load_dotenv(override=True)


def test_ark_chat() -> None:
    import os
    from langchain_ark.chat_models import ChatArk
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt_template = PromptTemplate.from_template("Hello, {role}")
    llm = ChatArk(model=os.environ["ARK_CHAT_MODEL"])
    parser = StrOutputParser()
    chain = prompt_template | llm | parser
    message = chain.invoke({"role": "Doubao"})
    assert isinstance(message, str)
