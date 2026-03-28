import { Bot, Zap } from 'lucide-react'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'

interface AgentCardProps {
  name: string
  description: string
  status: 'idle' | 'running' | 'error'
  lastActive: string | null
  turnsToday: number
  tokensToday: number
}

const statusConfig = {
  idle: { variant: 'neutral' as const, label: 'Idle' },
  running: { variant: 'success' as const, label: 'Running' },
  error: { variant: 'danger' as const, label: 'Error' },
}

export function AgentCard({ name, description, status, lastActive, turnsToday, tokensToday }: AgentCardProps) {
  const { variant, label } = statusConfig[status]

  return (
    <Card>
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-primary)]/10">
            <Bot size={20} className="text-[var(--color-primary)]" />
          </div>
          <div>
            <h3 className="font-semibold capitalize">{name}</h3>
            <p className="text-xs text-[var(--color-text-muted)]">{description}</p>
          </div>
        </div>
        <Badge variant={variant}>{label}</Badge>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-4 border-t border-[var(--color-border)] pt-4">
        <div>
          <p className="text-xs text-[var(--color-text-muted)]">Last active</p>
          <p className="text-sm font-medium">
            {lastActive ? new Date(lastActive).toLocaleString() : '—'}
          </p>
        </div>
        <div>
          <p className="text-xs text-[var(--color-text-muted)]">Turns today</p>
          <p className="text-sm font-medium">{turnsToday}</p>
        </div>
        <div className="flex items-start gap-1">
          <div>
            <p className="text-xs text-[var(--color-text-muted)]">Tokens today</p>
            <p className="text-sm font-medium">{tokensToday.toLocaleString()}</p>
          </div>
          <Zap size={12} className="mt-4 text-amber-400" />
        </div>
      </div>
    </Card>
  )
}
