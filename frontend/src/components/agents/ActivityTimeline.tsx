import { CheckCircle2, XCircle, Clock } from 'lucide-react'
import { useSupabase } from '../../hooks/useSupabase'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Spinner } from '../ui/Spinner'
import type { ScheduleLogEntry } from '../../types/database'

function formatTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function formatSkill(name: string): string {
  return name
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ')
}

const statusIcon = {
  done: <CheckCircle2 size={16} className="text-emerald-500" />,
  failed: <XCircle size={16} className="text-red-500" />,
}

export function ActivityTimeline() {
  const { data, loading, error } = useSupabase<ScheduleLogEntry>({
    table: 'ralf_schedule_log',
    order: { column: 'created_at', ascending: false },
    limit: 30,
  })

  if (loading) return <Spinner />
  if (error) return <p className="text-sm text-[var(--color-danger)]">{error}</p>

  if (data.length === 0) {
    return (
      <Card>
        <h3 className="mb-3 font-semibold">Recent Activity</h3>
        <p className="text-sm text-[var(--color-text-muted)]">No activity logged yet.</p>
      </Card>
    )
  }

  // Group by date
  const byDate: Record<string, ScheduleLogEntry[]> = {}
  for (const entry of data) {
    const date = entry.schedule_date ?? entry.created_at.slice(0, 10)
    if (!byDate[date]) byDate[date] = []
    byDate[date].push(entry)
  }

  return (
    <Card>
      <h3 className="mb-4 font-semibold">Recent Activity</h3>
      <div className="space-y-4">
        {Object.entries(byDate).map(([date, entries]) => (
          <div key={date}>
            <p className="mb-2 text-xs font-medium text-[var(--color-text-muted)]">
              {new Date(date + 'T00:00:00').toLocaleDateString(undefined, {
                weekday: 'short',
                month: 'short',
                day: 'numeric',
              })}
            </p>
            <div className="space-y-1.5">
              {entries.map((entry) => (
                <div
                  key={entry.id}
                  className="flex items-start gap-2.5 rounded-lg px-2 py-1.5 text-sm transition-colors hover:bg-[var(--color-surface-hover)]"
                >
                  <div className="mt-0.5">
                    {statusIcon[entry.status as keyof typeof statusIcon] ?? (
                      <Clock size={16} className="text-[var(--color-text-muted)]" />
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{formatSkill(entry.skill)}</span>
                      {entry.site && (
                        <Badge variant="neutral">{entry.site}</Badge>
                      )}
                    </div>
                    {entry.summary && (
                      <p className="mt-0.5 truncate text-xs text-[var(--color-text-muted)]">
                        {entry.summary}
                      </p>
                    )}
                  </div>
                  <span className="shrink-0 text-xs text-[var(--color-text-muted)]">
                    {formatTime(entry.completed_at ?? entry.created_at)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}
