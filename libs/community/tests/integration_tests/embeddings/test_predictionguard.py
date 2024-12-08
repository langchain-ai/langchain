"""Test Prediction Guard API wrapper"""

from langchain_community.embeddings.predictionguard import PredictionGuardEmbeddings


def test_predictionguard_embeddings_documents() -> None:
    """Test Prediction Guard embeddings."""
    embeddings = PredictionGuardEmbeddings(model="bridgetower-large-itm-mlm-itc")
    documents = [
        "embed this",
    ]
    output = embeddings.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 2


def test_predictionguard_embeddings_documents_multiple() -> None:
    """Test Prediction Guard embeddings."""
    embeddings = PredictionGuardEmbeddings(model="bridgetower-large-itm-mlm-itc")
    documents = [
        "embed me",
        "embed this",
    ]
    output = embeddings.embed_documents(documents)
    assert len(output[0]) > 2
    assert len(output[1]) > 2


def test_predictionguard_embeddings_query() -> None:
    """Test Prediction Guard embeddings."""
    embeddings = PredictionGuardEmbeddings(model="bridgetower-large-itm-mlm-itc")
    document = "embed this"
    output = embeddings.embed_query(document)
    assert len(output) > 2


def test_predictionguard_embeddings_images() -> None:
    """Test Prediction Guard embeddings."""
    embeddings = PredictionGuardEmbeddings(model="bridgetower-large-itm-mlm-itc")
    image = [
        "https://fastly.picsum.photos/id/866/200/300.jpg?hmac=rcadCENKh4rD6MAp6V_ma-AyWv641M4iiOpe1RyFHeI",
    ]
    output = embeddings.embed_images(image)
    assert len(output) == 1


def test_predictionguard_embeddings_images_multiple() -> None:
    """Test Prediction Guard embeddings."""
    embeddings = PredictionGuardEmbeddings(model="bridgetower-large-itm-mlm-itc")
    images = [
        "https://farm4.staticflickr.com/3300/3497460990_11dfb95dd1_z.jpg",
        "https://fastly.picsum.photos/id/866/200/300.jpg?hmac=rcadCENKh4rD6MAp6V_ma-AyWv641M4iiOpe1RyFHeI",
    ]
    output = embeddings.embed_images(images)
    assert len(output) == 2


def test_predictionguard_embeddings_image_text() -> None:
    """Test Prediction Guard Embeddings"""
    embeddings = PredictionGuardEmbeddings(model="bridgetower-large-itm-mlm-itc")
    inputs = [
        {
            "text": "embed me",
            "image": "https://fastly.picsum.photos/id/866/200/300.jpg?hmac=rcadCENKh4rD6MAp6V_ma-AyWv641M4iiOpe1RyFHeI",
        },
    ]
    output = embeddings.embed_image_text(inputs)
    assert len(output) == 1


def test_predictionguard_embeddings_image_text_multiple() -> None:
    """Test Prediction Guard Embeddings"""
    embeddings = PredictionGuardEmbeddings(model="bridgetower-large-itm-mlm-itc")
    inputs = [
        {
            "text": "embed me",
            "image": "https://fastly.picsum.photos/id/866/200/300.jpg?hmac=rcadCENKh4rD6MAp6V_ma-AyWv641M4iiOpe1RyFHeI",
        },
        {
            "text": "embed this",
            "image": "https://farm4.staticflickr.com/3300/3497460990_11dfb95dd1_z.jpg",
        },
    ]
    output = embeddings.embed_image_text(inputs)
    assert len(output) == 2
