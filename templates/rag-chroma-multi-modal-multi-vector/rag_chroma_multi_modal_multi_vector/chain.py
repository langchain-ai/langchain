import base64
import io
import os
from pathlib import Path

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.pydantic_v1 import BaseModel
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.schema.messages import HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import UpstashRedisByteStore
from langchain.vectorstores import Chroma
from PIL import Image


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string.

    :param base64_string: A Base64 encoded string of the image to be resized.
    :param size: A tuple representing the new size (width, height) for the image.
    :return: A Base64 encoded string of the resized image.
    """
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize(size, Image.LANCZOS)
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_resized_images(docs):
    """
    Resize images from base64-encoded strings.

    :param docs: A list of base64-encoded image to be resized.
    :return: Dict containing a list of resized base64-encoded strings.
    """
    b64_images = []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content
        resized_image = resize_base64_image(doc, size=(1280, 720))
        b64_images.append(resized_image)
    return {"images": b64_images}


def img_prompt_func(data_dict, num_images=2):
    """
    GPT-4V prompt for image analysis.

    :param data_dict: A dict with images and a user-provided question.
    :param num_images: Number of images to include in the prompt.
    :return: A list containing message objects for each image and the text prompt.
    """
    messages = []
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"][:num_images]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)
    text_message = {
        "type": "text",
        "text": (
            "You are an analyst tasked with answering questions about visual content.\n"
            "You will be give a set of image(s) from a slide deck / presentation.\n"
            "Use this information to answer the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


def multi_modal_rag_chain(retriever):
    """
    Multi-modal RAG chain,

    :param retriever: A function that retrieves the necessary context for the model.
    :return: A chain of functions representing the multi-modal RAG process.
    """
    # Initialize the multi-modal Large Language Model with specific parameters
    model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024)

    # Define the RAG pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(get_resized_images),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain


# Load chroma
vectorstore_mvr = Chroma(
    collection_name="image_summaries",
    persist_directory=str(Path(__file__).parent.parent / "chroma_db_multi_modal"),
    embedding_function=OpenAIEmbeddings(),
)

# Load redis
UPSTASH_URL = os.getenv("UPSTASH_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_TOKEN")
store = UpstashRedisByteStore(url=UPSTASH_URL, token=UPSTASH_TOKEN)
id_key = "doc_id"

# Create the multi-vector retriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore_mvr,
    byte_store=store,
    id_key=id_key,
)

# Create RAG chain
chain = multi_modal_rag_chain(retriever)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
