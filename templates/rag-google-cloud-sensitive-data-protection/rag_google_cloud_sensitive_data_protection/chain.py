import os
from typing import List, Tuple

from google.cloud import dlp_v2
from langchain_community.chat_models import ChatVertexAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableParallel


# Formatting for chat history
def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


def _deidentify_with_replace(
    input_str: str,
    info_types: List[str],
    project: str,
) -> str:
    """Uses the Data Loss Prevention API to deidentify sensitive data in a
    string by replacing matched input values with the info type.
    Args:
        project: The Google Cloud project id to use as a parent resource.
        input_str: The string to deidentify (will be treated as text).
        info_types: A list of strings representing info types to look for.
    Returns:
        str: The input string after it has been deidentified.
    """

    # Instantiate a client
    dlp = dlp_v2.DlpServiceClient()

    # Convert the project id into a full resource id.
    parent = f"projects/{project}/locations/global"

    if info_types is None:
        info_types = ["PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD_NUMBER"]
    # Construct inspect configuration dictionary
    inspect_config = {"info_types": [{"name": info_type} for info_type in info_types]}

    # Construct deidentify configuration dictionary
    deidentify_config = {
        "info_type_transformations": {
            "transformations": [
                {"primitive_transformation": {"replace_with_info_type_config": {}}}
            ]
        }
    }

    # Construct item
    item = {"value": input_str}

    # Call the API
    response = dlp.deidentify_content(
        request={
            "parent": parent,
            "deidentify_config": deidentify_config,
            "inspect_config": inspect_config,
            "item": item,
        }
    )

    # Print out the results.
    return response.item.value


# Prompt we will use
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant who translates to pirate",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

# Create Vertex AI retriever
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT_ID")
model_type = os.environ.get("MODEL_TYPE")

# Set LLM and embeddings
model = ChatVertexAI(model_name=model_type, temperature=0.0)


class ChatHistory(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})


_inputs = RunnableParallel(
    {
        "question": RunnableLambda(
            lambda x: _deidentify_with_replace(
                input_str=x["question"],
                info_types=["PERSON_NAME", "PHONE_NUMBER", "EMAIL_ADDRESS"],
                project=project_id,
            )
        ).with_config(run_name="<lambda> _deidentify_with_replace"),
        "chat_history": RunnableLambda(
            lambda x: _format_chat_history(x["chat_history"])
        ).with_config(run_name="<lambda> _format_chat_history"),
    }
)

# RAG
chain = _inputs | prompt | model | StrOutputParser()

chain = chain.with_types(input_type=ChatHistory).with_config(run_name="Inputs")
