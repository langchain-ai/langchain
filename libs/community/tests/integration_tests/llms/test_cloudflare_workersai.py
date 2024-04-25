import responses

from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI


@responses.activate
def test_cloudflare_workersai_call() -> None:
    responses.add(
        responses.POST,
        "https://api.cloudflare.com/client/v4/accounts/my_account_id/ai/run/@cf/meta/llama-2-7b-chat-int8",
        json={"result": {"response": "4"}},
        status=200,
    )

    llm = CloudflareWorkersAI(
        account_id="my_account_id",
        api_token="my_api_token",
        model="@cf/meta/llama-2-7b-chat-int8",
    )
    output = llm.invoke("What is 2 + 2?")

    assert output == "4"


@responses.activate
def test_cloudflare_workersai_stream() -> None:
    response_body = ['data: {"response": "Hello"}', "data: [DONE]"]
    responses.add(
        responses.POST,
        "https://api.cloudflare.com/client/v4/accounts/my_account_id/ai/run/@cf/meta/llama-2-7b-chat-int8",
        body="\n".join(response_body),
        status=200,
    )

    llm = CloudflareWorkersAI(
        account_id="my_account_id",
        api_token="my_api_token",
        model="@cf/meta/llama-2-7b-chat-int8",
        streaming=True,
    )

    outputs = []
    for chunk in llm.stream("Say Hello"):
        outputs.append(chunk)

    assert "".join(outputs) == "Hello"
