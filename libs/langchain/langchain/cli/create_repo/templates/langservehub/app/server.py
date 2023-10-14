from fastapi import FastAPI
from langserve import add_package_routes

app = FastAPI()

add_package_routes(app, "packages")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
