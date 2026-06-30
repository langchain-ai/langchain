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
import { Textarea } from "@/components/ui/textarea"
import { api, type PlanResponse } from "@/lib/api"

export function PlannerView() {
  const [goal, setGoal] = useState(
    "Plan a two-part blog series introducing LangChain agents.",
  )
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PlanResponse | null>(null)

  async function run() {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      setResult(await api.plan(goal))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Planner agent</CardTitle>
        <CardDescription>
          Uses <code>TodoListMiddleware</code> to break a goal into a plan.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Textarea
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          rows={3}
        />
        <Button onClick={run} disabled={loading || !goal.trim()}>
          {loading ? <Loader2 className="animate-spin" /> : "Plan"}
        </Button>
        {error && <p className="text-sm text-destructive">{error}</p>}
        {result && (
          <div className="space-y-3">
            {result.todos.length > 0 && (
              <ol className="space-y-1.5">
                {result.todos.map((t, i) => (
                  <li key={i} className="flex items-center gap-2 text-sm">
                    <Badge variant="outline">{t.status}</Badge>
                    {t.content}
                  </li>
                ))}
              </ol>
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
