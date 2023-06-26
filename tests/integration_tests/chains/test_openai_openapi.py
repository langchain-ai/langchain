from langchain.chains.openai_functions.openapi import get_openapi_chain


def test_openai_opeanapi() -> None:
    chain = get_openapi_chain(
        "https://www.klarna.com/us/shopping/public/openai/v0/api-docs/"
    )
    output = chain.run("What are some options for a men's large blue button down shirt")

    assert isinstance(output, str)


def test_openai_opeanapi_headers() -> None:
    headers = {"Authorization": "Bearer 3oRrWH4nYwHD0u9+wcdaNWHtvwxXVlXZxGF50+r2WWA="}
    chain = get_openapi_chain(
        "https://app.swaggerhub.com/apis/brandfetch/brandfetch/2.0.0", headers=headers
    )
    output = chain.run("What are some options for a men's large blue button down shirt")

    assert isinstance(output, str)
