import { useState } from "react"
import { Loader2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { api, type Person } from "@/lib/api"

export function ExtractorView() {
  const [text, setText] = useState(
    "Hi, I'm Ada Lovelace, I'm 36 years old, and you can reach me at ada@analytical-engine.org.",
  )
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [person, setPerson] = useState<Person | null>(null)

  async function run() {
    setLoading(true)
    setError(null)
    setPerson(null)
    try {
      setPerson(await api.extract(text))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Structured extractor</CardTitle>
        <CardDescription>
          Pulls a typed <code>Person</code> out of free text via{" "}
          <code>response_format</code>.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
        />
        <Button onClick={run} disabled={loading || !text.trim()}>
          {loading ? <Loader2 className="animate-spin" /> : "Extract"}
        </Button>
        {error && <p className="text-sm text-destructive">{error}</p>}
        {person && (
          <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 rounded-lg border bg-muted/40 p-4 text-sm">
            <span className="text-muted-foreground">name</span>
            <span className="font-medium">{person.name}</span>
            <span className="text-muted-foreground">age</span>
            <span className="font-medium">{person.age}</span>
            <span className="text-muted-foreground">email</span>
            <span className="font-medium">{person.email}</span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
