from pathlib import Path
from typing import Any

from langchain_experimental.open_clip import OpenCLIPEmbeddings
from PIL import Image

from langchain_chroma import Chroma


def create_temp_image(path: Path) -> None:
    """Create a small solid-color RGB image for testing purposes.

    Args:
        path (Path): Path to save the temporary image file.
    """
    img = Image.new("RGB", (64, 64), color=(127, 200, 80))
    img.save(path)


def test_openclip_chroma_embed_no_nesting_error(tmp_path: Path) -> None:
    """
    Regression test confirming that OpenCLIPEmbeddings + Chroma
    work correctly with image input, without external dependencies.

    This verifies that the triple-nested embedding bug no longer occurs
    and that similarity_search_by_image returns valid results.

    Args:
        tmp_path (Path): A pytest fixture providing a temporary directory.

    Returns:
        None
    """
    # Directory for isolated Chroma persistence
    chroma_dir: Path = tmp_path / "chroma_test_db"

    # Dynamically generate a sample image
    img_path: Path = tmp_path / "sample.jpg"
    create_temp_image(img_path)

    # Initialize OpenCLIP embeddings (1024-dimensional vectors)
    emb: OpenCLIPEmbeddings = OpenCLIPEmbeddings(
        model="ViT-B-32", pretrained="laion2b_s34b_b79k"
    )

    # Initialize Chroma vector store
    vs: Chroma = Chroma(
        collection_name="test_openclip_chroma",
        embedding_function=emb,
        persist_directory=str(chroma_dir),
    )

    # Add the generated image into Chroma
    vs.add_images(
        uris=[str(img_path)],
        metadatas=[{"id": "sample_image"}],
    )

    # Execute similarity search using the same image
    results: list[Any] = vs.similarity_search_by_image(uri=str(img_path))

    # Assertions to validate output
    assert isinstance(results, list)
    assert len(results) > 0, "Expected non-empty similarity search results"
    assert hasattr(results[0], "metadata"), "Each result should include metadata"
    assert results[0].metadata.get("id") == "sample_image"
