import os
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings

# Load images
img_dump_path = Path(__file__).parent / "docs/"
rel_img_dump_path = img_dump_path.relative_to(Path.cwd())
image_uris = sorted(
    [
        os.path.join(rel_img_dump_path, image_name)
        for image_name in os.listdir(rel_img_dump_path)
        if image_name.endswith(".jpg")
    ]
)

# Index
vectorstore = Path(__file__).parent / "chroma_db_multi_modal"
re_vectorstore_path = vectorstore.relative_to(Path.cwd())

# Load embedding function
print("Loading embedding function")  # noqa: T201
embedding = OpenCLIPEmbeddings(model_name="ViT-H-14", checkpoint="laion2b_s32b_b79k")

# Create chroma
vectorstore_mmembd = Chroma(
    collection_name="multi-modal-rag",
    persist_directory=str(Path(__file__).parent / "chroma_db_multi_modal"),
    embedding_function=embedding,
)

# Add images
print("Embedding images")  # noqa: T201
vectorstore_mmembd.add_images(uris=image_uris)
