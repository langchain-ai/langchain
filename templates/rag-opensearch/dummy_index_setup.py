import os

from openai import OpenAI
from opensearchpy import OpenSearch

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "https://localhost:9200")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "admin")
OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "langchain-test")

docs = [
    "[INFO] Initializing machine learning training job. Model: Convolutional Neural Network Dataset: MNIST Hyperparameters: \n   - Learning Rate: 0.001\n   - Batch Size: 64",
    "[INFO] Loading training data. Training data loaded successfully. Number of training samples: 60,000",
    "[INFO] Loading validation data. Validation data loaded successfully. Number of validation samples: 10,000",
    "[INFO] Training started. Epoch 1/10\n   - Loss: 0.532\n   - Accuracy: 0.812 Epoch 2/10\n   - Loss: 0.398\n   - Accuracy: 0.874 Epoch 3/10\n   - Loss: 0.325\n   - Accuracy: 0.901 ... (training progress) Training completed.",
    "[INFO] Validation started. Validation loss: 0.287 Validation accuracy: 0.915 Model performance meets validation criteria. Saving the model.",
    "[INFO] Testing the trained model. Test loss: 0.298 Test accuracy: 0.910",
    "[INFO] Deploying the trained model to production. Model deployment successful. API endpoint: http://your-api-endpoint/predict",
    "[INFO] Monitoring system initialized. Monitoring metrics:\n   - CPU Usage: 25%\n   - Memory Usage: 40%\n   - GPU Usage: 80%",
    "[ALERT] High GPU Usage Detected! Scaling resources to handle increased load.",
    "[INFO] Machine learning training job completed successfully. Total training time: 3 hours and 45 minutes.",
    "[INFO] Cleaning up resources. Job artifacts removed. Training environment closed."
    "[INFO] Image processing web server started. Listening on port 8080.",
    "[INFO] Received image processing request from client at IP address 192.168.1.100. Preprocessing image: resizing to 800x600 pixels. Image preprocessing completed successfully.",
    "[INFO] Applying filters to enhance image details. Filters applied: sharpening, contrast adjustment. Image enhancement completed.",
    "[INFO] Generating thumbnail for the processed image. Thumbnail generated successfully.",
    "[INFO] Uploading processed image to the user's gallery. Image successfully added to the gallery. Image ID: 123456.",
    "[INFO] Sending notification to the user: Image processing complete. Notification sent successfully.",
    "[ERROR] Failed to process image due to corrupted file format. Informing the client about the issue. Client notified about the image processing failure.",
    "[INFO] Image processing web server shutting down. Cleaning up resources. Server shutdown complete.",
]


client_oai = OpenAI(api_key=OPENAI_API_KEY)


client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
    use_ssl=True,
    verify_certs=False,
)

# Define the index settings and mappings
index_settings = {
    "settings": {
        "index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}
    },
    "mappings": {
        "properties": {
            "vector_field": {
                "type": "knn_vector",
                "dimension": 1536,
                "method": {"name": "hnsw", "space_type": "l2", "engine": "faiss"},
            }
        }
    },
}

response = client.indices.create(index=OPENSEARCH_INDEX_NAME, body=index_settings)

print(response)


# Insert docs


for each in docs:
    res = client_oai.embeddings.create(input=each, model="text-embedding-ada-002")

    document = {
        "vector_field": res.data[0].embedding,
        "text": each,
    }

    response = client.index(index=OPENSEARCH_INDEX_NAME, body=document, refresh=True)

    print(response)
