import { useEffect, useState } from "react"
import { Bot, Sparkles } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { api, type Health } from "@/lib/api"
import { ChatView } from "@/views/ChatView"
import { ExtractorView } from "@/views/ExtractorView"
import { PlannerView } from "@/views/PlannerView"
import { ReactAgentView } from "@/views/ReactAgentView"

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState(false)

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch(() => setHealthError(true))
  }, [])

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto max-w-3xl px-4 py-10">
        <header className="mb-8 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-xl font-semibold tracking-tight">
                LangChain v1 Playground
              </h1>
              <p className="text-sm text-muted-foreground">
                Four agents, one UI — powered by <code>create_agent</code>.
              </p>
            </div>
          </div>
          {healthError ? (
            <Badge variant="destructive">backend offline</Badge>
          ) : health ? (
            <Badge variant={health.openai_key ? "secondary" : "destructive"}>
              {health.openai_key ? "key ready" : "no API key"}
            </Badge>
          ) : (
            <Badge variant="outline">…</Badge>
          )}
        </header>

        <Tabs defaultValue="react">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="react">ReAct</TabsTrigger>
            <TabsTrigger value="extract">Extract</TabsTrigger>
            <TabsTrigger value="chat">Chat</TabsTrigger>
            <TabsTrigger value="plan">Planner</TabsTrigger>
          </TabsList>
          <TabsContent value="react">
            <ReactAgentView />
          </TabsContent>
          <TabsContent value="extract">
            <ExtractorView />
          </TabsContent>
          <TabsContent value="chat">
            <ChatView />
          </TabsContent>
          <TabsContent value="plan">
            <PlannerView />
          </TabsContent>
        </Tabs>

        <footer className="mt-10 flex items-center justify-center gap-2 text-xs text-muted-foreground">
          <Bot className="h-3 w-3" /> React + shadcn/ui · FastAPI · LangChain v1
        </footer>
      </div>
    </div>
  )
}
