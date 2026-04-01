import { useSupabase } from '../../hooks/useSupabase'
import type { AgentTurn } from '../../types/database'
import { Card, CardHeader, CardTitle } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Table } from '../ui/Table'
import { Spinner } from '../ui/Spinner'

const turnTypeVariant: Record<string, 'info' | 'success' | 'warning' | 'neutral'> = {
  user: 'info',
  cron: 'success',
  pulse: 'warning',
  worker: 'neutral',
}

const columns = [
  {
    key: 'created_at',
    header: 'Time',
    render: (row: AgentTurn) => (
      <span className="text-xs text-[var(--color-text-muted)]">
        {new Date(row.created_at).toLocaleString()}
      </span>
    ),
  },
  {
    key: 'agent_name',
    header: 'Agent',
    render: (row: AgentTurn) => <span className="font-medium capitalize">{row.agent_name}</span>,
  },
  {
    key: 'turn_type',
    header: 'Type',
    render: (row: AgentTurn) => (
      <Badge variant={turnTypeVariant[row.turn_type] ?? 'neutral'}>{row.turn_type}</Badge>
    ),
  },
  {
    key: 'input',
    header: 'Input',
    render: (row: AgentTurn) => (
      <span className="max-w-xs truncate text-xs">{row.input?.slice(0, 80) ?? '—'}</span>
    ),
    className: 'max-w-xs',
  },
  {
    key: 'model',
    header: 'Model',
    render: (row: AgentTurn) => (
      <span className="text-xs text-[var(--color-text-muted)]">{row.model ?? '—'}</span>
    ),
  },
  {
    key: 'tokens_used',
    header: 'Tokens',
    render: (row: AgentTurn) => <span className="text-xs">{row.tokens_used.toLocaleString()}</span>,
  },
  {
    key: 'duration_ms',
    header: 'Duration',
    render: (row: AgentTurn) => (
      <span className="text-xs">
        {row.duration_ms != null ? `${(row.duration_ms / 1000).toFixed(1)}s` : '—'}
      </span>
    ),
  },
]

export function AgentTurnLog() {
  const { data, loading, error } = useSupabase<AgentTurn>({
    table: 'agent_turns',
    order: { column: 'created_at', ascending: false },
    limit: 50,
    realtime: true,
  })

  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Agent Activity</CardTitle>
      </CardHeader>
      {loading ? (
        <Spinner />
      ) : error ? (
        <p className="text-sm text-[var(--color-danger)]">{error}</p>
      ) : (
        <Table columns={columns} data={data} />
      )}
    </Card>
  )
}
