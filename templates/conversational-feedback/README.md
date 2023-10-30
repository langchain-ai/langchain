# Chat Feedback Template

This template captures "implicit feedback from human behavior" within a simple chat bot. This instructs an LLM to use a user's responses within a conversation to grade its previous replies.

[Chat bots](https://python.langchain.com/docs/use_cases/chatbots) are perhaps the most common interface for applying LLMs. The quality of chat bots can be variable, and it can be challenging to get explicit user feedback through things like thumbs up/down buttons. Traditional analytics like "session length" or "conversation length" can be ambiguous. For multi-turn conversations with a chat bot, an immense amount of information can be gleaned from the dialog itself, which we can convert to metrics for monitoring, fine-tuning, and further evalution.

Taking [Chat Langchain](https://chat.langchain.com/) as a case study, only ~0.04% of all queries receive explicit feedback, but ~70% of of the queries are follow-ups to previous questions. A large portion of these followup queries
 are actual continuations of the same topic (vs. simply asking another question) and can be used to infer the quality of the previous AI response.

## LangSmith Feedback

[LangSmith](https://smith.langchain.com/) is a platform for building production-grade LLM applications. In addition to its debugging and offline evaluation functionality, it helps you capture feedback (both user and model-assisted) to improve your LLM application. For other cookbook examples on collecting feedback using LangSmith, check out the [docs](https://docs.smith.langchain.com/cookbook/feedback-examples).

## Implementation

The feedback is executed within a custom `RunEvaluator`. This evaluator is run in a separate thread (to not obstruct the runtime of the actual chat bot).
It uses an LLM (in this case, `gpt-3.5-turbo`) to grade the the AI's most recent chat message, accounting for the user's response to that message.

The prompt used within the LLM [can be found on the hub](https://smith.langchain.com/hub/wfh/response-effectiveness). Feel free to modify it to your liking!
We also use OpenAI's function calling API to ensure more consistent structured output in the grade.

## Environment Variables

Be sure that `OPENAI_API_KEY` is set in order to use the OpenAI models. Also, configure LangSmith by setting your `LANGSMITH_API_KEY`.

```bash
export OPENAI_API_KEY=sk-...
export LANGSMITH_API_KEY=...
```

## Usage

If you are deplying this via `LangServe`, it's recommended that you instruct the server to return callback events as well.

```
from conversational_feedback.chain import chain

add_routes(app, chain, path="/conversational-feedback", include_callback_events=True)
```

This will unify any traces started in the client with those in the server.

The code snippet below shows how you could use the stream endpoint

```python
from functools import partial
from typing import Dict, Optional, Callable, List
from langserve import RemoteRunnable
from langchain.callbacks.manager import tracing_v2_enabled
from langchain.schema import BaseMessage, AIMessage, HumanMessage

# Upudate with the URL provided by your LangServe server
chain = RemoteRunnable("http://127.0.0.1:8031/conversational-feedback")


def stream_content(
    text: str,
    chat_history: Optional[List[BaseMessage]] = None,
    last_run_id: Optional[str] = None,
    on_chunk=Callable,
):
    results = []
    with tracing_v2_enabled() as cb:
        for chunk in chain.stream(
            {"text": text, "chat_history": chat_history, "last_run_id": last_run_id},
        ):
            on_chunk(chunk)
            results.append(chunk)
        last_run_id = cb.latest_run.id if cb.latest_run is not None else None
    return last_run_id, "".join(results)


chat_history = []
text = "Where are my keys?"
last_run_id, response_message = stream_content(text, on_chunk=partial(print, end=""))
chat_history.extend([HumanMessage(content=text), AIMessage(content=response_message)])
text = "I CANT FIND THEM ANYWHERE" # The previous response will be given a low score since
# the user's frustration (as evidenced by the text) seems to be increasing
last_run_id, response_message = stream_content(
    text,
    chat_history=chat_history,
    last_run_id=str(last_run_id),
    on_chunk=partial(print, end=""),
)
chat_history.extend([HumanMessage(content=text), AIMessage(content=response_message)])
```
