# Querying the RAG Web Service

Start the FastAPI app and send a question using `curl`:

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -H "X-API-Token: <your token>" \
     -d '{"question": "What is RAG?"}'
```

Set `API_TOKEN` in the environment before launching the server to secure the endpoint.
