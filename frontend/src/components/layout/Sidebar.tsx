import { NavLink } from 'react-router-dom'
import { Bot, CalendarClock, Network, Users, Search, X } from 'lucide-react'

const links = [
  { to: '/agents', label: 'Agents', icon: Bot },
  { to: '/schedules', label: 'Schedules', icon: CalendarClock },
  { to: '/org-chart', label: 'Org Chart', icon: Network },
  { to: '/crm', label: 'CRM', icon: Users },
  { to: '/keywords', label: 'Keywords', icon: Search },
]

interface SidebarProps {
  open: boolean
  onClose: () => void
}

export function Sidebar({ open, onClose }: SidebarProps) {
  return (
    <>
      {/* Mobile overlay backdrop */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed left-0 top-0 z-50 flex h-screen w-56 flex-col border-r border-[var(--color-border)] bg-[var(--color-surface)] transition-transform duration-200 md:translate-x-0 ${
          open ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Logo + close button */}
        <div className="flex h-16 items-center justify-between border-b border-[var(--color-border)] px-5">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--color-primary)]">
              <Bot size={18} className="text-white" />
            </div>
            <span className="text-base font-semibold tracking-tight">Agent Hub</span>
          </div>
          <button
            onClick={onClose}
            className="rounded p-1 text-[var(--color-text-muted)] hover:bg-[var(--color-surface-hover)] md:hidden"
          >
            <X size={18} />
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 space-y-1 p-3">
          {links.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              onClick={onClose}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-[var(--color-primary)]/10 text-[var(--color-primary)]'
                    : 'text-[var(--color-text-muted)] hover:bg-[var(--color-surface-hover)] hover:text-[var(--color-text)]'
                }`
              }
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="border-t border-[var(--color-border)] px-5 py-4 text-xs text-[var(--color-text-muted)]">
          SEO Agent Dashboard
        </div>
      </aside>
    </>
  )
}
