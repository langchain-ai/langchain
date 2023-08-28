"""Test Titan Takeoff wrapper."""


import responses

from langchain.llms.titan_takeoff import TitanTakeoff


@responses.activate
def test_titan_takeoff_call() -> None:
    """Test valid call to Titan Takeoff."""
    url = "http://localhost:8000/generate"
    responses.add(responses.POST, url, json={"message": "2 + 2 is 4"}, status=200)

    # response = requests.post(url)
    llm = TitanTakeoff()
    output = llm("What is 2 + 2?")
    assert isinstance(output, str)
