import base64
import io
import os
from pathlib import Path

import pypdfium2 as pdfium
from langchain.chat_models import ChatOpenAI
from langchain.pydantic_v1 import BaseModel
from langchain.schema.document import Document
from langchain.schema.messages import HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from PIL import Image


def get_images_from_pdf(pdf_path, img_dump_path):
    """
    Extract images from each page of a PDF document and save as JPEG files.

    :param pdf_path: A string representing the path to the PDF file.
    :param img_dump_path: A string representing the path to dummp images.
    """
    pdf = pdfium.PdfDocument(pdf_path)
    n_pages = len(pdf)
    for page_number in range(n_pages):
        page = pdf.get_page(page_number)
        bitmap = page.render(scale=1, rotation=0, crop=(0, 0, 0, 0))
        pil_image = bitmap.to_pil()
        pil_image.save(f"{img_dump_path}/img_{page_number + 1}.jpg", format="JPEG")


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


# Input
# Current working directory is the parent directory of the docs subdirectory
doc_path = Path(__file__).parent / "docs/DDOG_Q3_earnings_deck.pdf"
img_dump_path = Path(__file__).parent / "docs/"
rel_doc_path = doc_path.relative_to(Path.cwd())
rel_img_dump_path = img_dump_path.relative_to(Path.cwd())
pil_images = get_images_from_pdf(rel_doc_path, rel_img_dump_path)

# Create chroma
vectorstore_mmembd = Chroma(
    collection_name="multi-modal-rag",
    embedding_function=OpenCLIPEmbeddings(
        model_name="ViT-H-14", checkpoint="laion2b_s32b_b79k"
    ),
)

# Get image URIs
image_uris = sorted(
    [
        os.path.join(rel_img_dump_path, image_name)
        for image_name in os.listdir(rel_img_dump_path)
        if image_name.endswith(".jpg")
    ]
)

# Add images
vectorstore_mmembd.add_images(uris=image_uris)

# Make retriever
retriever_mmembd = vectorstore_mmembd.as_retriever()

# Create RAG chain
chain = multi_modal_rag_chain(retriever_mmembd)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
