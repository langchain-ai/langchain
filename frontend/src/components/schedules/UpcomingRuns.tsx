import { Clock } from 'lucide-react'
import { useMemo } from 'react'
import { useSupabase } from '../../hooks/useSupabase'
import type { CronExecution, CronJob } from '../../types/database'
import { Card, CardHeader, CardTitle } from '../ui/Card'

interface Props {
  jobs: CronJob[]
}

export function UpcomingRuns({ jobs }: Props) {
  const { data: executions } = useSupabase<CronExecution>({
    table: 'cron_executions',
    order: { column: 'fired_at', ascending: false },
    limit: 10,
  })

  const upcoming = useMemo(() => {
    const runs: { job: string; time: Date }[] = []

    for (const job of jobs) {
      const lastExec = executions.find((e) => e.job_id === job.id)
      const lastFired = lastExec ? new Date(lastExec.fired_at) : new Date()
      const intervalMs = (job.interval_hours ?? 0) * 3600000 + (job.interval_minutes ?? 0) * 60000

      for (let i = 1; i <= 3; i++) {
        runs.push({ job: job.id, time: new Date(lastFired.getTime() + intervalMs * i) })
      }
    }

    return runs.sort((a, b) => a.time.getTime() - b.time.getTime()).slice(0, 8)
  }, [jobs, executions])

  return (
    <Card>
      <CardHeader>
        <CardTitle>Upcoming Runs</CardTitle>
      </CardHeader>
      {upcoming.length === 0 ? (
        <p className="text-sm text-[var(--color-text-muted)]">No scheduled runs</p>
      ) : (
        <div className="space-y-3">
          {upcoming.map((run, i) => (
            <div key={i} className="flex items-center gap-3">
              <Clock size={14} className="text-[var(--color-text-muted)]" />
              <div>
                <span className="text-sm font-medium capitalize">{run.job}</span>
                <p className="text-xs text-[var(--color-text-muted)]">
                  {run.time.toLocaleString()}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}
