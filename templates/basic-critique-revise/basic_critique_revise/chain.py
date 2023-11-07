import json
from datetime import datetime
from enum import Enum
from operator import itemgetter
from typing import Dict, Sequence

from langchain.chains.openai_functions import convert_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field, ValidationError
from langchain.schema.runnable import (
    ConfigurableField,
    Runnable,
    RunnableBranch,
    RunnableConfig,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)


class TaskType(str, Enum):
    call = "Call"
    message = "Message"
    todo = "Todo"
    in_person_meeting = "In-Person Meeting"
    email = "Email"
    mail = "Mail"
    text = "Text"
    open_house = "Open House"


class Task(BaseModel):
    title: str = Field(..., description="The title of the tasks, reminders and alerts")
    due_date: datetime = Field(
        ..., description="Due date. Must be a valid ISO date string with timezone"
    )
    task_type: TaskType = Field(None, description="The type of task")


class Tasks(BaseModel):
    """JSON definition for creating tasks, reminders and alerts"""

    tasks: Sequence[Task]


template = """Respond to the following user query to the best of your ability:

{query}"""

generate_prompt = ChatPromptTemplate.from_template(template)

function_args = {"functions": [convert_to_openai_function(Tasks)]}

task_function_call_model = ChatOpenAI(model="gpt-3.5-turbo").bind(**function_args)

output_parser = RunnableLambda(
    lambda x: json.loads(x.additional_kwargs["function_call"]["arguments"])
)


revise_template = """
Based on the provided context, fix the incorrect result of the original prompt
and the provided errors. Only respond with an answer that satisfies the
constraints laid out in the original prompt and fixes the Pydantic errors.

Hint: Datetime fields must be valid ISO date strings.

<context>
<original_prompt>
{original_prompt}
</original_prompt>
<incorrect_result>
{completion}
</incorrect_result>
<errors>
{error}
</errors>
</context>"""

revise_prompt = ChatPromptTemplate.from_template(revise_template)

revise_chain = revise_prompt | task_function_call_model | output_parser


def output_validator(output):
    try:
        Tasks.validate(output["completion"])
    except ValidationError as e:
        return str(e)

    return None


class IntermediateType(BaseModel):
    error: str
    completion: Dict
    original_prompt: str


validation_step = RunnablePassthrough().assign(error=RunnableLambda(output_validator))


def revise_loop(input: IntermediateType, config: RunnableConfig) -> IntermediateType:
    revise_step = RunnablePassthrough().assign(completion=revise_chain)

    else_step: Runnable[IntermediateType, IntermediateType] = RunnableBranch(
        (lambda x: x["error"] is None, RunnablePassthrough()),
        revise_step | validation_step,
    ).with_types(input_type=IntermediateType)

    max_iters = config.configurable.get("max_revisions", 5)  # WRONG
    for _ in range(max(0, max_iters - 1)):
        else_step = RunnableBranch(
            (lambda x: x["error"] is None, RunnablePassthrough()),
            revise_step | validation_step | else_step,
        )
    return else_step.invoke(input)


revise_lambda = RunnableLambda(revise_loop).configurable_fields(
    max_iterations=ConfigurableField(
        id="max_revisions",
        name="Max Revisions",
    )  # I think wrong?
)  # configurable_fields doesn't exist on lambda?


class InputType(BaseModel):
    query: str


chain = (
    generate_prompt
    | RunnableParallel(
        original_prompt=RunnablePassthrough(),
        completion=task_function_call_model | output_parser,
    )
    | validation_step
    | else_step
    | RunnableLambda(itemgetter("completion"))
).with_types(input_type=InputType)
