from fastapi import FastAPI
from langserve import add_routes

from ____project_name_identifier.chain import get_chain

app = FastAPI()

add_routes(
    app,
    get_chain(),
    config_keys=["tags"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
