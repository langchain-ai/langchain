from __future__ import annotations

from typing import List, Optional

from langchain import hub
from langchain.callbacks.tracers.evaluation import EvaluatorCallbackHandler
from langchain.callbacks.tracers.schemas import Run
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
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable
from langsmith.evaluation import EvaluationResult, RunEvaluator
from langsmith.schemas import Example

###############################################################################
# |   Chat Bot Evaluator Definition
# | This section defines an evaluator that evaluates any chat bot
# | without explicit user feedback. It formats the dialog up to
# | the current message and then instructs an LLM to grade the last AI response
# | based on the subsequent user response. If no chat history is present,
# V the evaluator is not called.
###############################################################################


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


def format_messages(input: dict) -> List[BaseMessage]:
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
    """Format messages and convert to a single string."""
    chat_history = format_messages(input)
    formatted_dialog = get_buffer_string(chat_history) + f"\nhuman: {input['text']}"
    return {"dialog": formatted_dialog}


def normalize_score(response: dict) -> dict:
    """Normalize the score to be between 0 and 1."""
    response["score"] = int(response["score"]) / 5
    return response


# To view the prompt in the playground: https://smith.langchain.com/hub/wfh/response-effectiveness
evaluation_prompt = hub.pull("wfh/response-effectiveness")
evaluate_response_effectiveness = (
    format_dialog
    | evaluation_prompt
    # bind_functions formats the schema for the OpenAI function
    # calling endpoint, which returns more reliable structured data.
    | ChatOpenAI(model="gpt-3.5-turbo").bind_functions(
        functions=[ResponseEffectiveness],
        function_call="ResponseEffectiveness",
    )
    # Convert the model's output to a dict
    | JsonOutputFunctionsParser(args_only=True)
    | normalize_score
)


class ResponseEffectivenessEvaluator(RunEvaluator):
    """Evaluate the chat bot based the subsequent user responses."""

    def __init__(self, evaluator_runnable: Runnable) -> None:
        super().__init__()
        self.runnable = evaluator_runnable

    def evaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> EvaluationResult:
        # This evaluator grades the AI's PREVIOUS response.
        # If no chat history is present, there isn't anything to evaluate
        # (it's the user's first message)
        if not run.inputs.get("chat_history"):
            return EvaluationResult(
                key="response_effectiveness", comment="No chat history present."
            )
        # This only occurs if the client isn't correctly sending the run IDs
        # of the previous calls.
        elif "last_run_id" not in run.inputs:
            return EvaluationResult(
                key="response_effectiveness", comment="No last run ID present."
            )
        # Call the LLM to evaluate the response
        eval_grade: Optional[dict] = self.runnable.invoke(run.inputs)
        target_run_id = run.inputs["last_run_id"]
        return EvaluationResult(
            **eval_grade,
            key="response_effectiveness",
            target_run_id=target_run_id,  # Requires langsmith >= 0.0.54
        )


###############################################################################
# |           The chat bot definition
# | This is what is actually exposed by LangServe in the API
# | It can be any chain that accepts the ChainInput schema and returns a str
# | all that is required is the with_config() call at the end to add the
# V evaluators as "listeners" to the chain.
# ############################################################################


class ChainInput(BaseModel):
    """Input for the chat bot."""

    chat_history: Optional[List[BaseMessage]] = Field(
        description="Previous chat messages."
    )
    text: str = Field(..., description="User's latest query.")
    last_run_id: Optional[str] = Field("", description="Run ID of the last run.")


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


def format_chat_history(chain_input: dict) -> dict:
    messages = format_messages(chain_input)

    return {
        "chat_history": messages,
        "text": chain_input.get("text"),
    }


# if you update the name of this, you MUST also update ../pyproject.toml
# with the new `tool.langserve.export_attr`
chain = (
    (format_chat_history | _prompt | _model | StrOutputParser())
    # This is to add the evaluators as "listeners"
    # and to customize the name of the chain.
    # Any chain that accepts a compatible input type works here.
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

chain = chain.with_types(input_type=ChainInput)
