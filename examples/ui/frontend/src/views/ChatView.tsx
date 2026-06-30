import { useRef, useState } from "react"
import { Loader2, Send } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { api } from "@/lib/api"

interface Msg {
  role: "user" | "bot"
  text: string
}

export function ChatView() {
  const threadId = useRef(`web-${Math.random().toString(36).slice(2)}`)
  const [messages, setMessages] = useState<Msg[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function send() {
    const text = input.trim()
    if (!text) return
    setInput("")
    setError(null)
    setMessages((m) => [...m, { role: "user", text }])
    setLoading(true)
    try {
      const res = await api.chat(threadId.current, text)
      setMessages((m) => [...m, { role: "bot", text: res.reply }])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Memory chatbot</CardTitle>
        <CardDescription>
          Remembers this conversation. Tell it your name, then ask it back.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex min-h-[220px] flex-col gap-2 rounded-lg border bg-muted/40 p-4">
          {messages.length === 0 && (
            <p className="text-sm text-muted-foreground">No messages yet.</p>
          )}
          {messages.map((m, i) => (
            <div
              key={i}
              className={m.role === "user" ? "self-end" : "self-start"}
            >
              <span
                className={
                  "inline-block max-w-[80%] rounded-2xl px-3 py-2 text-sm " +
                  (m.role === "user"
                    ? "bg-primary text-primary-foreground"
                    : "bg-secondary text-secondary-foreground")
                }
              >
                {m.text}
              </span>
            </div>
          ))}
          {loading && (
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          )}
        </div>
        {error && <p className="text-sm text-destructive">{error}</p>}
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") send()
            }}
            placeholder="Type a message…"
            disabled={loading}
          />
          <Button onClick={send} disabled={loading || !input.trim()} size="icon">
            <Send />
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
