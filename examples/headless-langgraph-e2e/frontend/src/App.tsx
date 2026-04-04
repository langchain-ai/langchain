import { useCallback, useMemo, useState, type ReactNode } from "react";
import { AIMessage, HumanMessage } from "langchain";
import {
  useStream,
  type DefaultToolCall,
  type ToolCallWithResult,
} from "@langchain/react";

import { LocationMap } from "./LocationMap";
import { browserNavigateClient, geolocationGetClient } from "./tools";

const ASSISTANT_ID = "agent";

const PRESETS = [
  "Please open https://example.com in the browser.",
  "Visit https://langchain.com and tell me the page title if you can.",
  "Where am I right now? Show me on a map.",
];

function parseResultContent(result: ToolCallWithResult<DefaultToolCall>["result"]): unknown {
  if (!result) return undefined;
  const c = result.content;
  const raw = typeof c === "string" ? c : JSON.stringify(c);
  try {
    return JSON.parse(raw) as unknown;
  } catch {
    return raw;
  }
}

function HeadlessToolCallCard({
  toolCall,
}: {
  toolCall: ToolCallWithResult<DefaultToolCall>;
}) {
  const { call, result, state } = toolCall;
  const pending = state === "pending";
  const parsed = parseResultContent(result);

  const title =
    call.name === "geolocation_get"
      ? "geolocation_get"
      : call.name === "browser_navigate"
        ? "browser_navigate"
        : call.name;

  const icon =
    call.name === "geolocation_get" ? "📍" : call.name === "browser_navigate" ? "🌐" : "🔧";

  let map: ReactNode = null;
  if (
    call.name === "geolocation_get" &&
    parsed &&
    typeof parsed === "object" &&
    parsed !== null &&
    "success" in parsed &&
    (parsed as { success?: boolean }).success === true &&
    "latitude" in parsed &&
    "longitude" in parsed
  ) {
    const p = parsed as { latitude: number; longitude: number; accuracy?: number };
    map = <LocationMap latitude={p.latitude} longitude={p.longitude} accuracy={p.accuracy} />;
  }

  let resultText = "";
  if (result) {
    const c = result.content;
    resultText = typeof c === "string" ? c : JSON.stringify(c);
  }

  return (
    <div className="tool-card">
      <header>
        <span aria-hidden="true">{icon}</span>
        <span>{pending ? `Running ${call.name}…` : title}</span>
      </header>
      <pre>{JSON.stringify(call.args, null, 2)}</pre>
      {map}
      {resultText ? (
        <pre>
          {map ? "---\n" : ""}
          {resultText}
        </pre>
      ) : null}
    </div>
  );
}

export function App() {
  const [apiUrl, setApiUrl] = useState("http://127.0.0.1:2024");

  const tools = useMemo(() => [browserNavigateClient, geolocationGetClient], []);

  const stream = useStream({
    apiUrl: apiUrl.replace(/\/$/, ""),
    assistantId: ASSISTANT_ID,
    tools,
  });

  const handleSubmit = useCallback(
    (text: string) => {
      void stream.submit({
        messages: [{ type: "human" as const, content: text }],
      });
    },
    [stream],
  );

  return (
    <>
      <h1>Headless tool (useStream)</h1>
      <p className="hint">
        Start the graph API from the parent directory:{" "}
        <code>uv run langgraph dev --config langgraph.json --no-browser</code>. Tool name{" "}
        <code>browser_navigate</code> and <code>geolocation_get</code> match <code>agent.py</code>
        ; the client runs implementations passed to <code>useStream</code> (see{" "}
        <code>ui-playground/.../headless-tools</code>). Geolocation uses OpenStreetMap embeds.
      </p>

      <div className="row">
        <div style={{ flex: 1, minWidth: "12rem" }}>
          <label htmlFor="apiUrl">LangGraph API base URL</label>
          <input
            id="apiUrl"
            type="url"
            value={apiUrl}
            onChange={(e) => {
              setApiUrl(e.target.value);
            }}
            autoComplete="off"
          />
        </div>
        {stream.messages.length > 0 ? (
          <button
            type="button"
            className="secondary"
            onClick={() => {
              stream.switchThread(null);
            }}
          >
            New thread
          </button>
        ) : null}
      </div>

      <label htmlFor="msg">Message</label>
      <textarea
        id="msg"
        rows={3}
        defaultValue={PRESETS[0]}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            handleSubmit((e.target as HTMLTextAreaElement).value);
          }
        }}
      />

      <button
        type="button"
        disabled={stream.isLoading}
        onClick={() => {
          const ta = document.getElementById("msg") as HTMLTextAreaElement | null;
          handleSubmit(ta?.value ?? "");
        }}
      >
        {stream.isLoading ? "Running…" : "Send"}
      </button>

      {stream.messages.length === 0 ? (
        <div style={{ marginTop: "1rem" }}>
          <span className="hint">Try a preset:</span>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginTop: "0.5rem" }}>
            {PRESETS.map((p) => (
              <button
                key={p}
                type="button"
                className="secondary"
                style={{ width: "auto" }}
                onClick={() => {
                  handleSubmit(p);
                }}
              >
                {p.slice(0, 42)}
                {p.length > 42 ? "…" : ""}
              </button>
            ))}
          </div>
        </div>
      ) : null}

      <div style={{ marginTop: "1.25rem" }}>
        {stream.messages.map((msg, idx) => {
          if (HumanMessage.isInstance(msg)) {
            return (
              <div key={msg.id ?? idx} className="bubble-user">
                {msg.text}
              </div>
            );
          }

          if (AIMessage.isInstance(msg)) {
            const msgToolCalls = stream.toolCalls.filter((tc) =>
              msg.tool_calls?.some((t) => t.id === tc.call.id),
            );

            if (msgToolCalls.length > 0) {
              return (
                <div key={msg.id ?? idx}>
                  {msgToolCalls.map((tc) => (
                    <HeadlessToolCallCard
                      key={tc.id}
                      toolCall={tc as ToolCallWithResult<DefaultToolCall>}
                    />
                  ))}
                </div>
              );
            }

            if (!msg.text) return null;
            return (
              <div key={msg.id ?? idx} className="bubble-ai">
                {msg.text}
              </div>
            );
          }

          return null;
        })}

        {stream.isLoading &&
          !stream.messages.some((m) => AIMessage.isInstance(m) && m.text) &&
          stream.toolCalls.length === 0 && <div className="typing">Thinking…</div>}

        {stream.error ? (
          <div className="error" role="alert">
            {stream.error instanceof Error ? stream.error.message : String(stream.error)}
          </div>
        ) : null}
      </div>
    </>
  );
}
