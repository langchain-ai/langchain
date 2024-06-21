imperative = [
    [
        "invoke",
        "str | List[dict | tuple | BaseMessage] | PromptValue",
        "BaseMessage",
        "A single chat model call.",
    ],
    [
        "ainvoke",
        "'''",
        "BaseMessage",
        "Defaults to running invoke in an async executor.",
    ],
    [
        "stream",
        "'''",
        "Iterator[BaseMessageChunk]",
        "Defaults to yielding output of invoke.",
    ],
    [
        "astream",
        "'''",
        "AsyncIterator[BaseMessageChunk]",
        "Defaults to yielding output of ainvoke.",
    ],
    [
        "astream_events",
        "'''",
        "AsyncIterator[StreamEvent]",
        "Event types: 'on_chat_model_start', 'on_chat_model_stream', 'on_chat_model_end'.",
    ],
    [
        "batch",
        "List[''']",
        "List[BaseMessage]",
        "Defaults to running invoke in concurrent threads.",
    ],
    [
        "abatch",
        "List[''']",
        "List[BaseMessage]",
        "Defaults to running ainvoke in concurrent threads.",
    ],
    [
        "batch_as_completed",
        "List[''']",
        "Iterator[Tuple[int, Union[BaseMessage, Exception]]]",
        "Defaults to running invoke in concurrent threads.",
    ],
    [
        "abatch_as_completed",
        "List[''']",
        "AsyncIterator[Tuple[int, Union[BaseMessage, Exception]]]",
        "Defaults to running ainvoke in concurrent threads.",
    ],
]
declarative = [
    [
        "bind_tools",
        # "Tools, ...",
        # "Runnable with same inputs/outputs as ChatModel",
        "Create ChatModel that can call tools.",
    ],
    [
        "with_structured_output",
        # "An output schema, ...",
        # "Runnable that takes ChatModel inputs and returns a dict or Pydantic object",
        "Create wrapper that structures model output using schema.",
    ],
    [
        "with_retry",
        # "Max retries, exceptions to handle, ...",
        # "Runnable with same inputs/outputs as ChatModel",
        "Create wrapper that retries model calls on failure.",
    ],
    [
        "with_fallbacks",
        # "List of models to fall back on",
        # "Runnable with same inputs/outputs as ChatModel",
        "Create wrapper that falls back to other models on failure.",
    ],
    [
        "configurable_fields",
        # "*ConfigurableField",
        # "Runnable with same inputs/outputs as ChatModel",
        "Specify init args of the model that can be configured at runtime via the RunnableConfig.",
    ],
    [
        "configurable_alternatives",
        # "ConfigurableField, ...",
        # "Runnable with same inputs/outputs as ChatModel",
        "Specify alternative models which can be swapped in at runtime via the RunnableConfig.",
    ],
]


def create_table(to_build: list) -> str:
    for x in to_build:
        x[0] = "`" + x[0] + "`"
    longest = [max(len(x[i]) for x in to_build) for i in range(len(to_build[0]))]
    widths = [int(1.2 * col) for col in longest]
    headers = (
        ["Method", "Input", "Output", "Description"]
        if len(widths) == 4
        else ["Method", "Description"]
    )
    rows = [[h + " " * (w - len(h)) for w, h in zip(widths, headers)]]
    for x in to_build:
        rows.append([y + " " * (w - len(y)) for w, y in zip(widths, x)])

    table = [" | ".join(([""] + x + [""])).strip() for x in rows]
    lines = [
        "+".join(([""] + ["-" * (len(y) + 2) for y in x] + [""])).strip() for x in rows
    ]
    lines[1] = lines[1].replace("-", "=")
    lines.append(lines[-1])
    rst = lines[0]
    for r, li in zip(table, lines[1:]):
        rst += "\n" + r + "\n" + li
    return rst
