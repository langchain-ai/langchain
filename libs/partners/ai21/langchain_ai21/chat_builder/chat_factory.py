from langchain_ai21.chat_builder.chat import (
    Chat,
    J2Chat,
    JambaChatCompletions,
)


def create_chat(model: str) -> Chat:
    if "j2" in model:
        return J2Chat()

    if "jamba" in model:
        return JambaChatCompletions()

    raise ValueError(f"Model {model} not supported.")
