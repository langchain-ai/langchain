import base64
import io
import os
import uuid
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.schema.messages import HumanMessage
from langchain.storage import UpstashRedisByteStore
from langchain.vectorstores import Chroma
from PIL import Image


def image_summarize(img_base64, prompt):
    """
    Make image summary

    :param img_base64: Base64 encoded string for image
    :param prompt: Text prompt for summarizatiomn
    :return: Image summarization prompt

    """
    chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content


def generate_img_summaries(img_base64_list):
    """
    Generate summaries for images

    :param img_base64_list: Base64 encoded images
    :return: List of image summaries and processed images
    """

    # Store image summaries
    image_summaries = []
    processed_images = []

    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""

    # Apply summarization to images
    for i, base64_image in enumerate(img_base64_list):
        try:
            image_summaries.append(image_summarize(base64_image, prompt))
            processed_images.append(base64_image)
        except Exception as e:
            print(f"Error with image {i+1}: {e}")

    return image_summaries, processed_images


def get_images_from_pdf(pdf_path):
    """
    Extract images from each page of a PDF document and save as JPEG files.

    :param pdf_path: A string representing the path to the PDF file.
    """
    pdf = pdfium.PdfDocument(pdf_path)
    n_pages = len(pdf)
    pil_images = []
    for page_number in range(n_pages):
        page = pdf.get_page(page_number)
        bitmap = page.render(scale=1, rotation=0, crop=(0, 0, 0, 0))
        pil_image = bitmap.to_pil()
        pil_images.append(pil_image)
    return pil_images


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string

    :param base64_string: Base64 string
    :param size: Image size
    :return: Re-sized Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    img_str = resize_base64_image(img_str, size=(960, 540))
    return img_str


def create_multi_vector_retriever(vectorstore, image_summaries, images):
    """
    Create retriever that indexes summaries, but returns raw images or texts

    :param vectorstore: Vectorstore to store embedded image sumamries
    :param image_summaries: Image summaries
    :param images: Base64 encoded images
    :return: Retriever
    """

    # Initialize the storage layer for images
    UPSTASH_URL = os.getenv("UPSTASH_URL")
    UPSTASH_TOKEN = os.getenv("UPSTASH_TOKEN")
    store = UpstashRedisByteStore(url=UPSTASH_URL, token=UPSTASH_TOKEN)
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )

    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    add_documents(retriever, image_summaries, images)

    return retriever


# Load PDF
doc_path = Path(__file__).parent / "docs/DDOG_Q3_earnings_deck.pdf"
rel_doc_path = doc_path.relative_to(Path.cwd())
print("Extract slides as images")
pil_images = get_images_from_pdf(rel_doc_path)

# Convert to b64
images_base_64 = [convert_to_base64(i) for i in pil_images]

# Image summaries
print("Generate image summaries")
image_summaries, images_base_64_processed = generate_img_summaries(images_base_64)

# The vectorstore to use to index the images summaries
vectorstore_mvr = Chroma(
    collection_name="image_summaries",
    persist_directory=str(Path(__file__).parent / "chroma_db_multi_modal"),
    embedding_function=OpenAIEmbeddings(),
)

# Create documents
images_base_64_processed_documents = [
    Document(page_content=i) for i in images_base_64_processed
]

# Create retriever
retriever_multi_vector_img = create_multi_vector_retriever(
    vectorstore_mvr,
    image_summaries,
    images_base_64_processed_documents,
)
