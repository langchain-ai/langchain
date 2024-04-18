from langchain_ai21.chat.chat_builder import ChatBuilder, J2ChatBuilder, JambaChatBuilder


def create_chat_builder(model: str) -> ChatBuilder:
    if "j2" in model:
        return J2ChatBuilder()

    if "jamba" in model:
        return JambaChatBuilder()

    raise ValueError(f"Model {model} not supported.")
