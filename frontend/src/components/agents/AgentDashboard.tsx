import { useMemo } from 'react'
import { useSupabase } from '../../hooks/useSupabase'
import type { AgentTurn, CronExecution } from '../../types/database'
import { AgentCard } from './AgentCard'
import { AgentTurnLog } from './AgentTurnLog'
import { CostChart } from './CostChart'
import { Spinner } from '../ui/Spinner'

const AGENTS = [
  { name: 'ralf', description: 'SEO agent — content, outreach, rankings, reporting' },
  { name: 'scraper', description: 'Web scraper — data collection and enrichment' },
]

export function AgentDashboard() {
  const todayStr = new Date().toISOString().slice(0, 10)

  const { data: turns, loading: loadingTurns } = useSupabase<AgentTurn>({
    table: 'agent_turns',
    order: { column: 'created_at', ascending: false },
    limit: 200,
  })

  const { data: executions, loading: loadingExecs } = useSupabase<CronExecution>({
    table: 'cron_executions',
    order: { column: 'fired_at', ascending: false },
    limit: 10,
  })

  if (loadingTurns || loadingExecs) {
    return <Spinner />
  }

  const agentStats = useMemo(() => {
    const stats: Record<string, { lastActive: string | null; turnsToday: number; tokensToday: number; hasRunning: boolean; hasError: boolean }> = {}

    for (const agent of AGENTS) {
      const agentTurns = turns.filter((t) => t.agent_name === agent.name)
      const todayTurns = agentTurns.filter((t) => t.created_at.startsWith(todayStr))

      stats[agent.name] = {
        lastActive: agentTurns[0]?.created_at ?? null,
        turnsToday: todayTurns.length,
        tokensToday: todayTurns.reduce((sum, t) => sum + t.tokens_used, 0),
        hasRunning: executions.some((e) => e.status === 'running'),
        hasError: executions.some((e) => e.status === 'failed'),
      }
    }
    return stats
  }, [turns, executions, todayStr])

  const getStatus = (name: string): 'idle' | 'running' | 'error' => {
    const s = agentStats[name]
    if (!s) return 'idle'
    if (s.hasRunning) return 'running'
    if (s.hasError) return 'error'
    return 'idle'
  }

  return (
    <div className="space-y-6">
      {/* Agent cards */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {AGENTS.map((agent) => (
          <AgentCard
            key={agent.name}
            name={agent.name}
            description={agent.description}
            status={getStatus(agent.name)}
            lastActive={agentStats[agent.name]?.lastActive ?? null}
            turnsToday={agentStats[agent.name]?.turnsToday ?? 0}
            tokensToday={agentStats[agent.name]?.tokensToday ?? 0}
          />
        ))}
      </div>

      {/* Cost chart */}
      <CostChart />

      {/* Turn log */}
      <AgentTurnLog />
    </div>
  )
}
