import { useState } from "react"
import { Loader2 } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { api, type ReactResponse } from "@/lib/api"

export function ReactAgentView() {
  const [question, setQuestion] = useState(
    "What's the weather in Tokyo, and what is 23 * 19?",
  )
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ReactResponse | null>(null)

  async function run() {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      setResult(await api.react(question))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>ReAct tool agent</CardTitle>
        <CardDescription>
          Calls <code>get_weather</code> and <code>calculator</code> tools to
          answer.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          <Input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") run()
            }}
            placeholder="Ask something…"
          />
          <Button onClick={run} disabled={loading || !question.trim()}>
            {loading ? <Loader2 className="animate-spin" /> : "Run"}
          </Button>
        </div>
        {error && <p className="text-sm text-destructive">{error}</p>}
        {result && (
          <div className="space-y-3">
            {result.tool_calls.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {result.tool_calls.map((t, i) => (
                  <Badge key={i} variant="secondary">
                    {t.tool} → {t.result}
                  </Badge>
                ))}
              </div>
            )}
            <div className="whitespace-pre-wrap rounded-lg border bg-muted/40 p-4 text-sm">
              {result.answer}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
