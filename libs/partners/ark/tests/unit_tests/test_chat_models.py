from dotenv import load_dotenv

load_dotenv(override=True)


def test_ark_chat() -> None:
    import os

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    from langchain_ark.chat_models import ChatArk

    prompt_template = PromptTemplate.from_template("Hello, {role}")
    llm = ChatArk(model=os.environ["ARK_CHAT_MODEL"])  # type: ignore[call-arg]
    parser = StrOutputParser()
    chain = prompt_template | llm | parser
    message = chain.invoke({"role": "Doubao"})
    assert isinstance(message, str)
