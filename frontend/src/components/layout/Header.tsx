import { useLocation } from 'react-router-dom'
import { RefreshCw } from 'lucide-react'
import { Button } from '../ui/Button'

const titles: Record<string, string> = {
  '/agents': 'Agents',
  '/schedules': 'Activity Schedules',
  '/org-chart': 'Org Chart',
  '/crm': 'CRM',
}

export function Header({ onRefresh }: { onRefresh?: () => void }) {
  const { pathname } = useLocation()
  const base = '/' + pathname.split('/')[1]
  const title = titles[base] ?? 'Dashboard'

  return (
    <header className="flex h-16 items-center justify-between border-b border-[var(--color-border)] px-6">
      <h1 className="text-xl font-semibold">{title}</h1>
      {onRefresh && (
        <Button variant="ghost" size="sm" onClick={onRefresh}>
          <RefreshCw size={14} className="mr-1.5" />
          Refresh
        </Button>
      )}
    </header>
  )
}
