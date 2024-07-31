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
class Usage(BaseModel):
    prompt_tokens: int = Field(0, description="The number of tokens in the prompt.")
    completion_tokens: int = Field(
        0, description="The number of tokens in the completion."
    )
    total_tokens: int = Field(
        0, description="The total number of tokens in the prompt and completion."
    )


# schema for messages used in the request as well as response
class Message(BaseModel):
    role: str = Field(..., description="The role of the message provider.")
    content: str = Field(..., description="The content of the message.")


# schema for the chat completion choices
class Choices(BaseModel):
    index: int = Field(..., description="Index of the choice.")
    message: Message = Field(..., description="Generated message's role and content.")
    finish_reason: str = Field(..., description="Finish reason for the generation.")


# request schema for generating chat completions
class GenerateRequest(BaseModel):
    model: str = Field(..., description="Model name used for the generation.")
    user: str = Field(
        ..., description="User name of the user who made the generation request."
    )
    messages: list[Message] = Field(
        ..., description="List of messages used for the generation."
    )


# response schema containing the generated response
class GeneratedResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the generation.")
    object: str = Field(
        ..., description="Name of the object type (should be 'chat.completion')."
    )
    created: int = Field(..., description="Timestamp when the generation was created.")
    model: str = Field(..., description="Model ID used for the generation.")
    choices: list[Choices] = Field(..., description="Generated text and finish reason.")
    usage: Usage = Field(..., description="Usage statistics for the generation.")


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
    async def health_check(self) -> JSONResponse:
        """
        Endpoint for health check.

        Returns:
            JSONResponse: A JSON response with status "ok".
        """
        return JSONResponse(status_code=200, content={"status": "ok"})

    # OpenAI compatible chat completion endpoint
    async def chat_completion(self, request: GenerateRequest) -> GeneratedResponse:
        """
        Endpoint for generating chat completions based on the provided request.

        Args:
            request (GenerateRequest): The request containing model, user, and messages.

        Returns:
            GeneratedResponse: The generated response including id, object, created timestamp, model, choices, and usage.
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
        message = Message(role="AI", content=generation["kwargs"]["content"])
        choices = Choices(index=0, message=message, finish_reason="stop")
        usage = Usage(
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
        res = GeneratedResponse(
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
