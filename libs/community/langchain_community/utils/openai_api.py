import logging
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse as JSONResponse
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable

logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d %(levelname)s:%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


# schema for token usage tracking returned as part of the response
class OpenAIAPIUsage(BaseModel):
    prompt_tokens: int = Field(
        0, description="The number of tokens in the user's prompt."
    )
    completion_tokens: int = Field(
        0, description="The number of tokens in the chat completion."
    )
    total_tokens: int = Field(
        0,
        description="The total number of tokens together in the prompt and the completion.",
    )


# schema for messages used in the request as well as response
class OpenAIAPIMessage(BaseModel):
    role: str = Field(..., description="The role of the message provider/generator.")
    content: str = Field(..., description="The content of the message.")


# schema for the chat completion choices
class OpenAIAPIChoices(BaseModel):
    index: int = Field(
        ..., description="0-indexed number of the choice in the response."
    )
    message: OpenAIAPIMessage = Field(
        ..., description="Generated message of type OpenAIAPIMessage."
    )
    finish_reason: str = Field(
        ..., description="The reason for finishing the chat completion."
    )


# request schema for generating chat completions
class ChatCompletionRequest(BaseModel):
    model: str = Field(
        ..., description="Name of the model to be used for the chat completion."
    )
    user: str = Field(
        ...,
        description="Name / human-readable ID of the user who made the chat completion request.",
    )
    messages: list[OpenAIAPIMessage] = Field(
        ..., description="List of messages used as input for the chat completion."
    )


# response schema containing the generated response
class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the chat completion.")
    object: str = Field(
        ..., description="Name of the object type (should be 'chat.completion')."
    )
    created: int = Field(
        ..., description="Timestamp for when the chat completion was created."
    )
    model: str = Field(..., description="Model ID used for the chat completion.")
    choices: list[OpenAIAPIChoices] = Field(
        ..., description="Generated text and finish reason."
    )
    usage: OpenAIAPIUsage = Field(
        ..., description="OpenAIAPIUsage statistics for the chat completion."
    )


class OpenAIChatCompletionAPI:
    def __init__(self, langchain_runnable: Runnable) -> None:
        """
        Initializes an OpenAI API compatible chat completions API endpoint
        for a provided LangChain runnable.

        Args:
            langchain_runnable (Runnable): The LangChain runnable to use for
            generating responses from your LangChain agent (system).

        Methods:
            health_check: Endpoint for health check, returns status "ok".
            chat_completion: Endpoint for generating chat completions based on the provided request.
            serve: Serve the FastAPI app using uvicorn.
        """

        self.app = FastAPI()
        self.langchain_runnable = langchain_runnable
        self.logger = logging.getLogger(__name__)
        self.logger.info("OpenAIChatCompletionAPI app initialized")
        # Register routes
        self.app.get("/")(self.health_check)
        self.app.post("/chat/completions")(self.chat_completion)

    # for health check
    async def liveness_check(self) -> JSONResponse:
        """
        Endpoint for health check.

        Returns:
            JSONResponse: A JSON response with status "ok".
        """
        return JSONResponse(status_code=200, content={"status": "ok"})

    # OpenAI compatible chat completion endpoint
    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Endpoint for generating chat completions based on the provided request.

        Args:
            request (ChatCompletionRequest): The request containing model, user, and messages.

        Returns:
            ChatCompletionResponse: The generated response including id, object, created timestamp, model, choices, and usage.
        """
        self.logger.info(f"Received request: {request}")
        self.logger.info(
            f"Generating response using LangChain runnable: {self.langchain_runnable}"
        )
        prompt = request.messages[-1].content
        generation = self.langchain_runnable.invoke(input=prompt).to_json()
        timestamp = int(datetime.now().timestamp())
        res_id = generation["kwargs"]["id"]
        model = self.langchain_runnable.model_name
        message = OpenAIAPIMessage(role="AI", content=generation["kwargs"]["content"])
        choices = OpenAIAPIChoices(index=0, message=message, finish_reason="stop")
        usage = OpenAIAPIUsage(
            prompt_tokens=generation["kwargs"]["response_metadata"]["usage_metadata"][
                "prompt_token_count"
            ],
            completion_tokens=generation["kwargs"]["response_metadata"][
                "usage_metadata"
            ]["candidates_token_count"],
            total_tokens=generation["kwargs"]["response_metadata"]["usage_metadata"][
                "total_token_count"
            ],
        )
        res = ChatCompletionResponse(
            id=res_id,
            object="chat.completion",
            created=timestamp,
            model=model,
            choices=[choices],
            usage=usage,
        )
        return res

    # serve the FastAPI app
    def serve(self, base_url: str = "0.0.0.0", port: int = 8080, **kwargs) -> None:
        """
        Serve the FastAPI app using uvicorn.

        Args:
            base_url (str, optional): The base URL for the API. Defaults to "0.0.0.0".
            port (int, optional): The port for the API. Defaults to 8080.
            **kwargs: Additional keyword arguments to pass to uvicorn.run().
        """
        uvicorn.run(self.app, host=base_url, port=port, **kwargs)
