import { AlertTriangle } from 'lucide-react'
import { isConfigured } from '../../utils/supabase'

export function ConfigBanner() {
  if (isConfigured) return null

  return (
    <div className="flex items-center gap-2 border-b border-[var(--color-danger)]/30 bg-[var(--color-danger)]/10 px-4 py-2 text-sm text-[var(--color-danger)]">
      <AlertTriangle size={16} className="shrink-0" />
      <span>
        Supabase not configured. Add{' '}
        <code className="rounded bg-[var(--color-surface)] px-1 py-0.5 text-xs">VITE_SUPABASE_URL</code>{' '}
        and{' '}
        <code className="rounded bg-[var(--color-surface)] px-1 py-0.5 text-xs">VITE_SUPABASE_ANON_KEY</code>{' '}
        to <code className="rounded bg-[var(--color-surface)] px-1 py-0.5 text-xs">.env.local</code>
      </span>
    </div>
  )
}
