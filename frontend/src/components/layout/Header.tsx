import { useLocation } from 'react-router-dom'
import { Menu, RefreshCw } from 'lucide-react'
import { Button } from '../ui/Button'

const titles: Record<string, string> = {
  '/agents': 'Agents',
  '/schedules': 'Activity Schedules',
  '/org-chart': 'Org Chart',
  '/crm': 'CRM',
  '/keywords': 'Keywords',
}

interface HeaderProps {
  onRefresh?: () => void
  onMenuToggle: () => void
}

export function Header({ onRefresh, onMenuToggle }: HeaderProps) {
  const { pathname } = useLocation()
  const base = '/' + pathname.split('/')[1]
  const title = titles[base] ?? 'Dashboard'

  return (
    <header className="flex h-16 items-center justify-between border-b border-[var(--color-border)] px-4 md:px-6">
      <div className="flex items-center gap-3">
        <button
          onClick={onMenuToggle}
          className="rounded-lg p-2 text-[var(--color-text-muted)] hover:bg-[var(--color-surface-hover)] md:hidden"
        >
          <Menu size={20} />
        </button>
        <h1 className="text-lg font-semibold md:text-xl">{title}</h1>
      </div>
      {onRefresh && (
        <Button variant="ghost" size="sm" onClick={onRefresh}>
          <RefreshCw size={14} className="mr-1.5" />
          Refresh
        </Button>
      )}
    </header>
  )
}
