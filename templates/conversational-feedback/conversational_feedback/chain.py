from __future__ import annotations

from typing import List, Optional

from langchain import hub
from langchain.callbacks.tracers.evaluation import EvaluatorCallbackHandler
from langchain.callbacks.tracers.schemas import Run
from langchain.chains.openai_functions.base import convert_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    StrOutputParser,
    get_buffer_string,
)
from langchain.schema.runnable import Runnable
from langsmith.evaluation import EvaluationResult, RunEvaluator
from langsmith.schemas import Example, Run

from pydantic import BaseModel, Field

### The feedback model used for the "function definition" provided to OpenAI
# For use with open source models, you can add the schema directly,
# but some modifications to the prompt and parser will be needed



class ResponseEffectiveness(BaseModel):
    """Score the effectiveness of the AI chat bot response."""

    reasoning: str = Field(
        ...,
        description="Explanation for the score.",
    )
    score: int = Field(
        ...,
        min=0,
        max=5,
        description="Effectiveness of AI's final response.",
    )


def format_messages(input: dict) -> dict:
    """Format the messages for the evaluator."""
    chat_history = input.get("chat_history") or []
    results = []
    for message in chat_history:
        if message["type"] == "human":
            results.append(HumanMessage.parse_obj(message))
        else:
            results.append(AIMessage.parse_obj(message))
    return results


def format_dialog(input: dict) -> dict:
    """Format the dialog for the evaluator."""
    chat_history = format_messages(input)
    formatted_dialog = get_buffer_string(chat_history)  # + f"\nhuman: {input['text']}"
    return {"dialog": formatted_dialog}


def normalize_score(response: dict) -> dict:
    """Normalize the score to be between 0 and 1."""
    response["score"] = int(response["score"]) / 5
    return response


evaluation_prompt = hub.pull("wfh/response-effectiveness")
evaluate_response_effectiveness = (
    # format_messages is a function that takes a dict and returns a dict
    format_dialog
    | evaluation_prompt
    # bind() provides the requested schemas to the model for structured prediction
    | ChatOpenAI(model="gpt-3.5-turbo").bind(
        functions=[convert_to_openai_function(ResponseEffectiveness)],
        function_name="response_effectiveness",
    )
    # Convert the model's output to a dict
    | JsonOutputFunctionsParser(args_only=True)
    | normalize_score
)


class ResponseEffectivenessEvaluator(RunEvaluator):
    def __init__(self, evaluator_runnable: Runnable) -> None:
        super().__init__()
        self.runnable = evaluator_runnable

    def evaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> EvaluationResult:
        # This particular evaluator is configured to evaluate the previous
        # AI response. It uses the user's followup question or comment as
        # additional grounding for its grade.
        if not run.inputs.get("chat_history"):
            return EvaluationResult(
                key="response_effectiveness", comment="No chat history present."
            )
        elif "last_run_id" not in run.inputs:
            return EvaluationResult(
                key="response_effectiveness", comment="No last run ID present."
            )
        eval_grade: Optional[dict] = self.runnable.invoke(run.inputs)
        target_run_id = run.inputs["last_run_id"]
        return EvaluationResult(
            **eval_grade,
            key="response_effectiveness",
            target_run_id=target_run_id,
        )


### The actual deployed chain (we are keeping it simple for this example)
# The main focus of this template is the evaluator above, not the chain itself.


class ChainInput(BaseModel):
    chat_history: Optional[List[BaseMessage]] = Field(
        description="Previous chat messages."
    )
    text: str = Field(..., description="User's latest query.")
    last_run_id: Optional[str] = Field("", description="ID of the last run.")


_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant who speaks like a pirate",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{text}"),
    ]
)
_model = ChatOpenAI()


def format_chat_history(chain_input: ChainInput) -> dict:
    # This is a hack to get the chat history into the prompt
    messages = format_messages(chain_input)

    return {
        "chat_history": messages,
        "text": chain_input.get("text"),
    }


# if you update the name of this, you MUST also update ../pyproject.toml
# with the new `tool.langserve.export_attr`
chain = (
    (format_chat_history | _prompt | _model | StrOutputParser())
    # This is to populate the openapi spec for LangServe
    .with_types(input_type=ChainInput)
    # This is to add the evluators as "listeners"
    # and to customize the name of the chain
    .with_config(
        run_name="ChatBot",
        callbacks=[
            EvaluatorCallbackHandler(
                evaluators=[
                    ResponseEffectivenessEvaluator(evaluate_response_effectiveness)
                ]
            )
        ],
    )
)
