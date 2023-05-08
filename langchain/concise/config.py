from langchain.base_language import BaseLanguageModel
from langchain.chat_models.openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter

__MAX_TOKENS: int = None
__DEFAULT_MODEL: BaseLanguageModel = None
__DEFAULT_TEXT_SPLITTER: TextSplitter = RecursiveCharacterTextSplitter(
    chunk_size=__MAX_TOKENS // 4,
    chunk_overlap=20,
    length_function=len,
)


def get_default_max_tokens() -> int:
    if __MAX_TOKENS is None:
        # has not yet been set, default to the models max tokens
        try:
            return get_default_model().max_tokens
        except AttributeError:
            # model doesn't have a max tokens attribute, default to 100
            return 100
    return __MAX_TOKENS


def get_default_model() -> BaseLanguageModel:
    if __DEFAULT_MODEL is None:
        # has not yet been set, default to ChatOpenAI('gpt-3.5-turbo')
        __DEFAULT_MODEL = ChatOpenAI("gpt-3.5-turbo")
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
