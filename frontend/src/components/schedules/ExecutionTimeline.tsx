import { CheckCircle, XCircle, Loader, Clock } from 'lucide-react'
import { useSupabase } from '../../hooks/useSupabase'
import type { CronExecution } from '../../types/database'
import { Card, CardHeader, CardTitle } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Spinner } from '../ui/Spinner'

const statusIcon: Record<string, React.ReactNode> = {
  completed: <CheckCircle size={16} className="text-emerald-400" />,
  failed: <XCircle size={16} className="text-red-400" />,
  running: <Loader size={16} className="animate-spin text-blue-400" />,
}

export function ExecutionTimeline() {
  const { data, loading, error } = useSupabase<CronExecution>({
    table: 'cron_executions',
    order: { column: 'fired_at', ascending: false },
    limit: 30,
  })

  return (
    <Card>
      <CardHeader>
        <CardTitle>Execution History</CardTitle>
      </CardHeader>
      {loading ? (
        <Spinner />
      ) : error ? (
        <p className="text-sm text-[var(--color-danger)]">{error}</p>
      ) : data.length === 0 ? (
        <p className="py-8 text-center text-sm text-[var(--color-text-muted)]">No executions yet</p>
      ) : (
        <div className="space-y-0">
          {data.map((exec, i) => {
            const duration =
              exec.completed_at && exec.fired_at
                ? ((new Date(exec.completed_at).getTime() - new Date(exec.fired_at).getTime()) / 1000).toFixed(0)
                : null

            return (
              <div key={exec.id} className="relative flex gap-4 pb-6">
                {/* Timeline line */}
                {i < data.length - 1 && (
                  <div className="absolute left-[19px] top-8 h-full w-px bg-[var(--color-border)]" />
                )}
                {/* Icon */}
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-[var(--color-border)] bg-[var(--color-bg)]">
                  {statusIcon[exec.status] ?? <Clock size={16} className="text-[var(--color-text-muted)]" />}
                </div>
                {/* Content */}
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium capitalize">{exec.job_id}</span>
                    <Badge
                      variant={
                        exec.status === 'completed' ? 'success' : exec.status === 'failed' ? 'danger' : 'info'
                      }
                    >
                      {exec.status}
                    </Badge>
                  </div>
                  <div className="mt-1 flex flex-wrap gap-x-4 text-xs text-[var(--color-text-muted)]">
                    <span>{new Date(exec.fired_at).toLocaleString()}</span>
                    {duration && <span>{duration}s</span>}
                    <span>{exec.tasks_executed ?? 0} tasks</span>
                    <span>{(exec.tokens_used ?? 0).toLocaleString()} tokens</span>
                  </div>
                  {exec.error && (
                    <p className="mt-1 rounded bg-red-500/10 px-2 py-1 text-xs text-red-400">
                      {exec.error.slice(0, 200)}
                    </p>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </Card>
  )
}
