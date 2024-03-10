import math
from bisect import bisect
from operator import itemgetter
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from langchain_core.language_models import BaseChatModel, LanguageModelLike
from langchain_core.outputs import LLMResult
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableLambda

from langchain.output_parsers import JsonOutputKeyToolsParser

_DEFAULT_OPENAI_TOOLS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Classify the user input.{class_descriptions}"),
        ("human", "{input}"),
    ]
)

_DEFAULT_OPENAI_LOGPROBS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify the user input.{class_descriptions} MAKE SURE your output is one of the classes and NOTHING else.",  # noqa: E501
        ),
        ("human", "{input}"),
    ]
)


def create_classification_chain(
    llm: LanguageModelLike,
    classes: Union[Sequence[str], Dict[str, str]],
    /,
    *,
    type: Literal["openai-tools", "openai-logprobs"],
    prompt: Optional[BasePromptTemplate] = None,
    **kwargs: Any,
) -> Runnable:
    """"""
    if not classes:
        raise ValueError("classes cannot be empty.")
    if type == "openai-tools":
        return create_openai_tool_classification_chain(
            llm, classes, prompt=prompt, **kwargs
        )
    elif type == "openai-logprobs":
        return create_openai_logprobs_classification_chain(
            llm, classes, prompt=prompt, **kwargs
        )
    # TODO:  Add JSON and XML chains.
    else:
        raise ValueError(
            f"Unknown type {type}. Expected one of 'openai-tools', 'openai-logprobs'."
        )


def create_openai_tool_classification_chain(
    llm: LanguageModelLike,
    classes: Union[Sequence[str], Dict[str, str]],
    /,
    *,
    prompt: Optional[BasePromptTemplate] = None,
) -> Runnable[Dict, str]:
    """"""
    prompt = prompt or _DEFAULT_OPENAI_TOOLS_PROMPT
    if isinstance(classes, Dict):
        descriptions = "\n".join(f"{k}: {v}" for k, v in classes.items())
        class_descriptions = f"\n\nThe classes are:\n\n{descriptions}"
    else:
        class_descriptions = ""
    if "class_descriptions" in prompt.input_variables:
        prompt = prompt.partial(class_descriptions=class_descriptions)
    class_names = ", ".join(f'"{c}"' for c in classes)
    tool = {
        "type": "function",
        "function": {
            "name": "classify",
            "description": "Classify the input as one of the given classes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "classification": {
                        "description": (
                            "The classification of the input. Must be one of "
                            f"{class_names}."
                        ),
                        "enum": list(classes),
                        "type": "string",
                    }
                },
                "required": ["classification"],
            },
        },
    }

    llm_with_tool = llm.bind(
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "classify"}},
    )
    return (
        prompt
        | llm_with_tool
        | JsonOutputKeyToolsParser(key="classify", return_single=True)
        | itemgetter("classification")
    )


def _parse_logprobs(
    result: LLMResult, classes: List[str], top_k: int
) -> Union[Dict, List]:
    original_classes = classes.copy()
    classes = [c.lower() for c in classes]
    top_classes = [c for c in classes if c in result.generations[0][0].text.lower()]

    logprobs = result.generations[0][0].generation_info["logprobs"]["content"]
    all_logprobs = [lp for token in logprobs for lp in token["top_logprobs"]]
    present_token_classes = [
        lp for lp in all_logprobs if lp["token"].strip().lower() in classes
    ]
    if not top_classes and not present_token_classes:
        res = {"classification": None, "confidence": None}
        return res if top_k == 1 else [res]

    # If any individual token matches a class.
    cumulative = {}
    for lp in present_token_classes:
        normalized = lp["token"].strip().lower()
        if normalized in cumulative:
            cumulative[normalized] += math.exp(lp["logprob"])
        else:
            cumulative[normalized] = math.exp(lp["logprob"])

    # If there are present classes that span more than a token.
    present_multi_token_classes = set(top_classes).difference(cumulative)
    spans = [len(logprobs[0]["token"])]
    for lp in logprobs[1:]:
        spans.append(len(lp["token"]))
    for top_class in present_multi_token_classes:
        start = result.generations[0][0].text.find(top_class)
        start_token_idx = bisect.bisect(spans, start)
        end = start + len(top_class)
        end_token_idx = bisect.bisect_left(spans, end)
        cumulative[top_class] = math.exp(
            sum(lp["logprob"] for lp in logprobs[start_token_idx : end_token_idx + 1])
        )
    res = sorted(
        [
            {"classification": original_classes[classes.index(k)], "confidence": v}
            for k, v in cumulative.items()
        ],
        key=(lambda x: x["confidence"]),
        reverse=True,
    )
    return res[0] if top_k == 1 else res[:top_k]


def create_openai_logprobs_classification_chain(
    llm: BaseChatModel,
    classes: Union[Sequence[str], Dict[str, str]],
    /,
    *,
    prompt: Optional[BasePromptTemplate] = None,
    top_k: int = 1,
) -> Runnable[Dict, Dict]:
    """"""
    prompt = prompt or _DEFAULT_OPENAI_TOOLS_PROMPT
    if isinstance(classes, Dict):
        descriptions = "\n".join(f"{k}: {v}" for k, v in classes.items())
        class_descriptions = f"\n\nThe classes are:\n\n{descriptions}\n\n"
    else:
        names = ", ".join(classes)
        class_descriptions = f"The classes are: {names}."
    prompt = prompt.partial(class_descriptions=class_descriptions)
    generate = RunnableLambda(llm.generate_prompt, afunc=llm.agenerate_prompt).bind(
        logprobs=True, top_logprobs=top_k
    )
    parse = RunnableLambda(_parse_logprobs).bind(classes=list(classes), top_k=top_k)
    return prompt | (lambda x: [x]) | generate | parse
