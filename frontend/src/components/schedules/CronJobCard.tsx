import { CalendarClock, Clock } from 'lucide-react'
import { useSupabase } from '../../hooks/useSupabase'
import type { CronExecution, CronJob } from '../../types/database'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'

interface Props {
  job: CronJob
}

export function CronJobCard({ job }: Props) {
  const { data: executions } = useSupabase<CronExecution>({
    table: 'cron_executions',
    filters: { job_id: job.id },
    order: { column: 'fired_at', ascending: false },
    limit: 20,
  })

  const latest = executions[0] ?? null
  const successRate =
    executions.length > 0
      ? Math.round((executions.filter((e) => e.status === 'completed').length / executions.length) * 100)
      : null

  const interval = job.interval_hours
    ? `Every ${job.interval_hours}h`
    : `Every ${job.interval_minutes}min`

  const statusVariant =
    latest?.status === 'completed' ? 'success' : latest?.status === 'running' ? 'info' : latest?.status === 'failed' ? 'danger' : 'neutral'

  return (
    <Card>
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald-500/10">
            <CalendarClock size={20} className="text-emerald-400" />
          </div>
          <div>
            <h3 className="font-semibold capitalize">{job.id}</h3>
            <p className="text-xs text-[var(--color-text-muted)]">{job.description}</p>
          </div>
        </div>
        <Badge variant={statusVariant}>{latest?.status ?? 'never run'}</Badge>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-4 border-t border-[var(--color-border)] pt-4">
        <div>
          <p className="text-xs text-[var(--color-text-muted)]">Interval</p>
          <div className="flex items-center gap-1">
            <Clock size={12} className="text-[var(--color-text-muted)]" />
            <p className="text-sm font-medium">{interval}</p>
          </div>
        </div>
        <div>
          <p className="text-xs text-[var(--color-text-muted)]">Last run</p>
          <p className="text-sm font-medium">
            {latest ? new Date(latest.fired_at).toLocaleString() : '—'}
          </p>
        </div>
        <div>
          <p className="text-xs text-[var(--color-text-muted)]">Success rate</p>
          <p className="text-sm font-medium">
            {successRate != null ? `${successRate}%` : '—'}
          </p>
        </div>
      </div>
    </Card>
  )
}
