from langchain.base_language import BaseLanguageModel
from langchain.chat_models.openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter

__MAX_TOKENS = 4096
__DEFAULT_MODEL = ChatOpenAI()
__DEFAULT_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=__MAX_TOKENS // 4,
    chunk_overlap=20,
    length_function=len,
)


def get_default_max_tokens() -> int:
    return __MAX_TOKENS


def get_default_model() -> BaseLanguageModel:
    return __DEFAULT_MODEL


def get_default_text_splitter() -> TextSplitter:
    return __DEFAULT_TEXT_SPLITTER


def set_default_max_tokens(max_tokens: int):
    global __MAX_TOKENS
    __MAX_TOKENS = max_tokens


def set_default_model(model: BaseLanguageModel):
    global __DEFAULT_MODEL
    __DEFAULT_MODEL = model


def set_default_text_splitter(splitter: TextSplitter):
    global __DEFAULT_TEXT_SPLITTER
    __DEFAULT_TEXT_SPLITTER = splitter
