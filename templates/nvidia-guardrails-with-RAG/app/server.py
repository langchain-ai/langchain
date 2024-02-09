from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from nvidia_rag_canonical import chain_with_guardrails as nvidia_rag_canonical_chain
from nvidia_rag_canonical import ingest as nvidia_rag_ingest

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

print()
# Edit this to add the chain you want to add
add_routes(app, nvidia_rag_canonical_chain, path="/nvidia-rag-canonical")
add_routes(app, nvidia_rag_ingest, path="/nvidia-rag-ingest")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
