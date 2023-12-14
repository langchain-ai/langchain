import os
from pathlib import Path

import pypdfium2 as pdfium
from langchain.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings


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


# Load PDF
doc_path = Path(__file__).parent / "docs/DDOG_Q3_earnings_deck.pdf"
img_dump_path = Path(__file__).parent / "docs/"
rel_doc_path = doc_path.relative_to(Path.cwd())
rel_img_dump_path = img_dump_path.relative_to(Path.cwd())
print("pdf index")
pil_images = get_images_from_pdf(rel_doc_path, rel_img_dump_path)
print("done")
vectorstore = Path(__file__).parent / "chroma_db_multi_modal"
re_vectorstore_path = vectorstore.relative_to(Path.cwd())

# Load embedding function
print("Loading embedding function")
embedding = OpenCLIPEmbeddings(model_name="ViT-H-14", checkpoint="laion2b_s32b_b79k")

# Create chroma
vectorstore_mmembd = Chroma(
    collection_name="multi-modal-rag",
    persist_directory=str(Path(__file__).parent / "chroma_db_multi_modal"),
    embedding_function=embedding,
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
print("Embedding images")
vectorstore_mmembd.add_images(uris=image_uris)
