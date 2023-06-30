import logging
import os
import time
from multiprocessing import Process
from typing import List

import pytest
import uvicorn
from fastapi import Body, FastAPI
from pydantic import BaseModel

from langchain.llms import SelfHostedApi

# class SmallInput(BaseModel):
#     prompt: str

app_no_schema = FastAPI()


@app_no_schema.post("/")
def root(prompt: str = Body(..., embed=True)):
    return {"response": f"Processed: {prompt}"}


class InputSchema(BaseModel):
    prompt: str
    temp: float = 0.7
    max_new_tokens: int = 200


class OutputSchema(BaseModel):
    response: str
    response_time: float
    initial_prompt: str
    max_new_tokens: int


app_with_schema = FastAPI()


@app_with_schema.post("/")
def root(input: InputSchema) -> OutputSchema:
    return OutputSchema(
        response=f"Processed at temp {input.temp}: {input.prompt}",
        response_time=0.1,
        initial_prompt=input.prompt,
        max_new_tokens=input.max_new_tokens,
    )


class NestedOutputSchema(BaseModel):
    input_schema: InputSchema
    output_schema: List[OutputSchema]


app_with_nested_schema = FastAPI()


@app_with_nested_schema.post("/")
def root(input: InputSchema) -> NestedOutputSchema:
    return NestedOutputSchema(
        input_schema=input,
        output_schema=[
            OutputSchema(
                response=f"Processed at temp {input.temp}: {input.prompt}",
                response_time=0.1,
                initial_prompt=input.prompt,
                max_new_tokens=input.max_new_tokens,
            )
        ],
    )


@pytest.fixture(scope="session")
def start_api_no_schema():
    server_working_dir = os.path.dirname(__file__)
    proc = Process(
        target=uvicorn.run,
        args=("test_self_hosted_api:app_no_schema",),
        kwargs={
            "port": 5050,
            "host": "localhost",
            "log_level": "info",
            "app_dir": server_working_dir,
        },
        daemon=True,
    )
    proc.start()
    time.sleep(0.1)
    return "http://localhost:5050/"


@pytest.fixture(scope="session")
def start_api_with_schema():
    server_working_dir = os.path.dirname(__file__)
    proc = Process(
        target=uvicorn.run,
        args=("test_self_hosted_api:app_with_schema",),
        kwargs={
            "port": 5051,
            "host": "localhost",
            "log_level": "info",
            "app_dir": server_working_dir,
        },
        daemon=True,
    )
    proc.start()
    time.sleep(0.1)
    return "http://localhost:5051/"


@pytest.fixture(scope="session")
def start_api_with_nested_schema():
    server_working_dir = os.path.dirname(__file__)
    proc = Process(
        target=uvicorn.run,
        args=("test_self_hosted_api:app_with_nested_schema",),
        kwargs={
            "port": 5052,
            "host": "localhost",
            "log_level": "info",
            "app_dir": server_working_dir,
        },
        daemon=True,
    )
    proc.start()
    # uvicorn.run(
    #             'test_self_hosted_api:app_with_nested_schema',
    #             **{
    #                 "port": 5052,
    #                 'host': 'localhost',
    #                 'log_level': 'info',
    #                 'app_dir': server_working_dir,
    #             },
    #             )
    time.sleep(0.5)
    return "http://localhost:5052/"


def test_happy_no_schema(start_api_no_schema):
    llm = SelfHostedApi(
        endpoint_url=start_api_no_schema,
        task="text-generation",
    )
    output = llm._call(prompt="Test prompt")
    assert output == "Processed: Test prompt"


def test_wrong_prompt_key_no_schema(start_api_no_schema):
    llm = SelfHostedApi(
        endpoint_url=start_api_no_schema,
        task="text-generation",
        prompt_key="wrong_key",
    )
    with pytest.raises(ValueError):
        output = llm._call(prompt="Test prompt")


def test_wrong_response_key_no_schema(start_api_no_schema):
    llm = SelfHostedApi(
        endpoint_url=start_api_no_schema,
        task="text-generation",
        prompt_key="prompt",
        response_key="wrong_key",
    )
    with pytest.raises(ValueError):
        output = llm._call(prompt="Test prompt")


def test_happy_with_schema_and_model_kwargs(start_api_with_schema):
    llm = SelfHostedApi(
        endpoint_url=start_api_with_schema,
        task="text-generation",
        model_kwargs={"temp": 0.2, "max_new_tokens": 100},
        input_schema=InputSchema,
        output_schema=OutputSchema,
        prompt_key="prompt",
        response_key="response",
    )

    output = llm._call(prompt="Test prompt")
    assert output == "Processed at temp 0.2: Test prompt"


def test_happy_with_schema_and_call_kwargs(start_api_with_schema):
    llm = SelfHostedApi(
        endpoint_url=start_api_with_schema,
        task="text-generation",
        input_schema=InputSchema,
        output_schema=OutputSchema,
        prompt_key="prompt",
        response_key="response",
    )

    output = llm._call(prompt="Test prompt", temp=0.2, max_new_tokens=100)
    assert output == "Processed at temp 0.2: Test prompt"


def test_model_kwargs_not_in_input_schema(start_api_with_schema):
    with pytest.raises(ValueError):
        llm = SelfHostedApi(
            endpoint_url=start_api_with_schema,
            task="text-generation",
            model_kwargs={"wrong_key": "string"},
            input_schema=InputSchema,
            output_schema=OutputSchema,
            prompt_key="prompt",
            response_key="response",
        )


def test_prompt_key_not_in_input_schema(start_api_with_schema):
    with pytest.raises(ValueError):
        llm = SelfHostedApi(
            endpoint_url=start_api_with_schema,
            task="text-generation",
            model_kwargs={"temp": 0.2, "max_new_tokens": 100},
            input_schema=InputSchema,
            output_schema=OutputSchema,
            prompt_key="wrong_key",
            response_key="response",
        )


def test_response_key_not_in_output_schema(start_api_with_schema):
    llm = SelfHostedApi(
        endpoint_url=start_api_with_schema,
        task="text-generation",
        model_kwargs={"temp": 0.2, "max_new_tokens": 100},
        input_schema=InputSchema,
        output_schema=OutputSchema,
        prompt_key="prompt",
        response_key="wrong_key",
    )
    with pytest.raises(ValueError):
        llm._call(prompt="Test prompt")


def test_nested_response_key(start_api_with_nested_schema):
    llm = SelfHostedApi(
        endpoint_url=start_api_with_nested_schema,
        task="text-generation",
        model_kwargs={"temp": 0.2, "max_new_tokens": 100},
        input_schema=InputSchema,
        output_schema=NestedOutputSchema,
        prompt_key="prompt",
        response_key=("output_schema", 0, "response"),
    )

    output = llm._call(prompt="Test prompt")
    assert output == "Processed at temp 0.2: Test prompt"


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://localhost:5055/",
        "http://localhost:5050/wrong",
    ],
)
def test_endpoint_dne(endpoint):
    llm = SelfHostedApi(
        endpoint_url=endpoint,
        task="text-generation",
        model_kwargs={"temp": 0.2, "max_new_tokens": 100},
        input_schema=InputSchema,
        output_schema=OutputSchema,
        prompt_key="prompt",
        response_key="response",
    )
    with pytest.raises(ValueError):
        llm._call(prompt="Test prompt")


def test_task_invalid(start_api_with_schema):
    with pytest.raises(ValueError):
        llm = SelfHostedApi(
            endpoint_url=start_api_with_schema,
            task="wrong-task",
            model_kwargs={"temp": 0.2, "max_new_tokens": 100},
            input_schema=InputSchema,
            output_schema=OutputSchema,
            prompt_key="prompt",
            response_key="response",
        )


def model_kwargs_no_input_schema(start_api_no_schema):
    with pytest.raises(ValueError):
        llm = SelfHostedApi(
            endpoint_url=start_api_no_schema,
            task="text-generation",
            model_kwargs={"temp": 0.2, "max_new_tokens": 100},
            prompt_key="prompt",
        )


# def test_no_response_key_with_output_schema(start_api_with_schema):
#     llm = SelfHostedApi(endpoint_url=start_api_with_schema,
#                         task='text-generation',
#                         model_kwargs={'temp': 0.2, 'max_new_tokens': 100},
#                         input_schema=InputSchema,
#                         output_schema=OutputSchema,
#                         prompt_key='prompt',
#                         )
#     with pytest.raises(ValueError):
#         llm._call(prompt='Test prompt')


def test_enforce_stop_tokens(start_api_no_schema):
    llm = SelfHostedApi(
        endpoint_url=start_api_no_schema,
        task="text-generation",
    )
    output = llm._call(prompt="Test <|end|> prompt", stop=["<|end|>"])
    assert output == "Processed: Test "
