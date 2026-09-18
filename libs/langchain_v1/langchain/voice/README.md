# LangChain Voice

LangChain Voice turns a LangGraph or Deep Agent into a complete duplex voice agent.
You define the graph and customer-facing personality; LangChain Voice owns the live
model connection, conversation loop, interruptions, and background work.

```python
from langchain.voice import OpenAIRealtimeConversationLayer, create_voice_agent

agent = create_voice_agent(
    graph=my_compiled_langgraph,
    conversation=OpenAIRealtimeConversationLayer(
        instructions="Be concise, conversational, and helpful.",
        model="gpt-realtime-2",
        voice="alloy",
    ),
)

await agent.run(transport=my_voice_transport)
```

Applications do not register coordination tools, dispatch function calls,
track task IDs, schedule responses, or manipulate provider events. LangChain Voice
does that internally and gives the customer one coherent assistant.

Completed task results use when-idle delivery. If the assistant is speaking,
LangChain Voice generates a follow-up after the heard response finishes. If the user
is speaking, the result is folded into the response to that user turn. Gemini
Live receives the final result as the continuation of its original non-blocking
function call. OpenAI Realtime receives bounded runtime context as a system
conversation item, never as text attributed to the user.

The transport is a typed duplex media boundary, so the same agent can use local
audio, browser audio, telephony, or LiveKit without putting device or room
details into the application brain.

For an existing local microphone and speaker implementation, use the included
compatibility adapter:

```python
from langchain.voice import LocalAudioTransport

transport = LocalAudioTransport(my_microphone, my_speaker)
await agent.run(transport=transport)
```

`agent.run(audio_in=my_microphone, audio_out=my_speaker)` remains supported as
a migration shortcut and constructs the same adapter internally.

For a connected LiveKit room, install `livekit` and pass its room transport:

```python
from langchain.voice import LiveKitAudioTransport

await room_job_context.connect()
transport = LiveKitAudioTransport(room_job_context.room)
await agent.run(transport=transport)
```

LangChain Voice owns the transport adapter and provider session; the surrounding
LiveKit job continues to own the room connection.

This is the same separation LangGraph uses: a compiled graph is directly
executable in-process; a deployment adds persistence, queues, workers, and
network APIs around it.

## What LangChain Voice manages

```text
audio -> OpenAI Realtime or Gemini Live -> internal work coordination
                    |                         |
                    |                 LangGraph thread(s)
                    |                         |
transport <- speakable updates    <- task events/results
```

Internally, the conversation layer owns what the user actually said and heard.
It does not put partially generated or interrupted speech into a graph
checkpoint. LangChain Voice keeps independent objectives isolated, runs them in
parallel, and restarts the appropriate graph thread when the user corrects a
request. That means "Actually, Heathrow only" can revise the flight search
without rewriting the voice transcript or stopping an unrelated hotel search.
A task represents one coherent objective rather than one utterance. Follow-up
work on that objective updates the same task and LangGraph thread; independently
deliverable work gets another task and may run in parallel. The conversation
agent remains responsible for relaying every useful result to the customer.

The conversation layer owns the voice persona instructions. LangChain Voice prepends
its private conversation-coordination rules, then gives the live model that
complete prompt and the framework-owned create, update, and cancel tools. The
graph remains responsible for its own reasoning prompt.

To use Gemini Live instead, configure the other conversation layer:

```python
from langchain.voice import GeminiLiveConversationLayer

conversation = GeminiLiveConversationLayer(
    instructions="Be concise, conversational, and helpful.",
    model="gemini-3.1-flash-live-preview",
    voice="Aoede",
)
```

## Install

LangChain Voice follows LangChain's Python requirement of Python 3.10 or newer.

```bash
pip install -U langchain
```

Choose the live conversation provider you want:

```bash
pip install 'langchain[voice-openai]'
pip install 'langchain[voice-gemini]'
```

The runtime accepts any compiled graph with an async `ainvoke` method.

The lower-level JSON WebSocket message/event adapter remains optional. This is
a text and task-control protocol; it is not an audio WebSocket transport:

```bash
pip install 'langchain[voice-websocket]'
```

## Try the browser voice demo

```bash
pip install -e '.[voice-websocket]'
python examples/voice/demo.py
```

Then open <http://127.0.0.1:8000>. The demo uses the browser's speech
recognition and speech synthesis, so it needs no API keys. Chrome-based browsers
currently provide the best Web Speech API support. Start two ordinary requests
to see parallel tasks; start a follow-up with "actually", "instead", or "only"
to make the demo conversation layer update its latest active task.

The demo sends final transcript text over the WebSocket. This is intentional:
it exercises conversation/task interruption semantics before committing the
framework to an audio codec or speech provider. Binary audio frames are
reserved for a future `VoiceTransport` implementation.

## Custom task inputs and results

The defaults call the graph like this:

```python
await graph.ainvoke(
    {"messages": [{"role": "user", "content": instruction}]},
    config={"configurable": {"thread_id": thread_id}},
)
```

For another state schema, provide adapters:

```python
agent = create_voice_agent(
    graph,
    conversation=OpenAIRealtimeConversationLayer(instructions="Keep acknowledgements short."),
    input_factory=lambda instruction: {"request": instruction},
    result_formatter=lambda state: state["answer"],
)
```

The default formatter understands strings, `response`, `output`, `result`, and
the final non-empty item in a LangGraph `messages` list.

## LangSmith traces

When LangSmith tracing is configured, every background task revision appears as
a `langchain_voice_task` child run under the live conversation trace. `create_task`
still returns immediately with its task ID and `started` status; the separate
task run stays open for the real graph work and ends with the final result,
failure, or cancellation. Updates close the superseded revision as cancelled
and create a new run for the next revision.

Task trace inputs and results are bounded, and graph failures use the same safe
customer-facing error as the task event. If LangSmith is not installed or its
tracing hook fails, task execution continues unchanged.

## Coordination diagnostics

Set `LANGCHAIN_VOICE_DEBUG=1` to print metadata-only diagnostics for Gemini task
coordination:

```bash
LANGCHAIN_VOICE_DEBUG=1 uv run voice-demo --backend langvoice
```

The logs correlate Gemini function-call IDs with LangChain Voice task IDs and show
terminal events, queue depth, result batches, generation triggers, and model
turn completion. Task instructions, result bodies, audio, and credentials are
not logged.

## Advanced: custom conversation layers

OpenAI Realtime and Gemini Live are the normal conversation layers and require
no application coordination code. Framework authors adding another multimodal
provider can implement the `ConversationLayer.run(...)` protocol. A layer owns
its provider connection and audio/text streaming; LangChain Voice continues to own
the graph runtime, task tools, and task-result coordination.

The provider interface intentionally stays small: configured persona
`instructions` plus one asynchronous `run(...)` method. Each run receives an
isolated `VoiceSession`, one `VoiceTransport`, an optional status UI, and a tracing
project name. Shared coordination code validates the same three task tools,
executes them, bounds task results, and relays terminal events. Provider adapters
only translate that contract into native events—for example continuing Gemini
function responses or OpenAI Realtime system context and response scheduling.

`VoiceTransport` owns media lifecycle and playout semantics. It yields typed
`AudioReceived` and `AudioPlayed` events, accepts self-describing `AudioFrame`
objects, reports whether output is active, waits for playout to drain, and returns
a `PlaybackReceipt` when output is interrupted. That receipt identifies the
provider stream and exact heard duration, allowing Realtime context truncation
without coupling the provider to a local speaker queue. A LiveKit adapter can map
these operations to room tracks and `AudioSource.clear_queue()`; a browser audio
adapter can map them to binary media plus client playout acknowledgements.

The optional text WebSocket adapter has a separate
`WebSocketServer(..., conversation_factory=...)` extension point. That factory
creates isolated text-adapter state for each connection and is not part of
`create_voice_agent`.

Conversation layers can call `VoiceSession.record_transcript(role, text)` to
record final user or assistant text. The realtime model remains in charge of
its turn loop while LangChain Voice exposes an authoritative transcript event stream.

## Advanced: in-process runtime

`VoiceAgent.run()` is the normal public boundary. Framework and deployment
authors can use `VoiceSession` directly to build another provider or transport:

```python
session = agent._create_session()

task_id = await session.send({"type": "task.create", "instruction": "Research Paris"})
await session.send(
    {
        "type": "task.update",
        "task_id": task_id,
        "instruction": "Research Paris and Berlin",
    }
)
await session.send({"type": "task.cancel", "task_id": task_id})

async for event in session.events():
    ...
```

`_create_session(send_event=callback)` is an internal extension point for
provider and deployment adapters. Events are always available through the
bounded async stream, so a deployment can apply backpressure rather than
accumulating unbounded session output. Application code should use `run()`.

## Optional WebSocket transport

The CLI is an adapter, not part of `VoiceAgent`:

```bash
python -m langchain.voice app:agent --host 127.0.0.1 --port 8765
```

Connect to `ws://127.0.0.1:8765`. The server first emits:

```json
{"type":"session.ready","session_id":"...","protocol_version":"0.2"}
```

Client messages for an explicitly configured custom text adapter:

```json
{"type":"input.text","text":"Find flights to London"}
{"type":"session.close"}
```

Low-level task-control messages remain available to provider and deployment
authors, but ordinary clients only send conversation input.

Server events include `conversation.transcript`, `conversation.message`,
`task.created`, `task.started`, `task.updated`, `task.completed`,
`task.cancelled`, `task.failed`, and `error`.

## Deployment model and safety

A production "LangSmith Voice Deployments" layer would own authentication,
durable session/task metadata, reconnects, queues, workers, autoscaling, and
SSE/WebSocket/LiveKit endpoints. LangChain Voice owns the executable session and task
semantics beneath that layer.

The development server binds to loopback by default. Before exposing it to the
internet:

- terminate TLS and authenticate connections at a trusted proxy;
- pass an explicit `origins=[...]` allowlist;
- keep tenant identity outside user-controlled JSON;
- replace in-memory session state if reconnect/resume is required;
- make sure the graph implementation propagates cancellation to remote runs.

Inputs, results, message size, queue depth, timeouts, and concurrent tasks are
bounded. Task IDs are scoped to one session, and closing a session cancels its
active work.

## Status

LangChain Voice is an alpha. OpenAI Realtime and Gemini Live audio conversations work
through the transport-neutral PCM interfaces. Resumable sessions,
authentication hooks, telephony, observability, and LiveKit integration are
next.
