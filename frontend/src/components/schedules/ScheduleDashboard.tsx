import { CronJobCard } from './CronJobCard'
import { ExecutionTimeline } from './ExecutionTimeline'
import { SkillSchedulePanel } from './SkillSchedulePanel'
import { UpcomingRuns } from './UpcomingRuns'
import type { CronJob } from '../../types/database'

const CRON_JOBS: CronJob[] = [
  {
    id: 'worker',
    interval_hours: 3,
    first_delay_seconds: 600,
    description: 'Heavy background tasks: content writing, keyword research, prospect enrichment, blog publishing',
  },
  {
    id: 'pulse',
    interval_minutes: 60,
    first_delay_seconds: 300,
    description: 'Lightweight check-in: ranking movers, budget alerts, blocker escalation, progress summary',
  },
]

export function ScheduleDashboard() {
  return (
    <div className="space-y-6">
      {/* Job cards */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {CRON_JOBS.map((job) => (
          <CronJobCard key={job.id} job={job} />
        ))}
      </div>

      {/* Skill schedule with descriptions and frequency controls */}
      <SkillSchedulePanel />

      {/* Upcoming + Timeline side by side */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-1">
          <UpcomingRuns jobs={CRON_JOBS} />
        </div>
        <div className="lg:col-span-2">
          <ExecutionTimeline />
        </div>
      </div>
    </div>
  )
}
