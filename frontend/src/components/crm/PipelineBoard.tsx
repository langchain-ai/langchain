import { useSupabase } from '../../hooks/useSupabase'
import type { CrmContact } from '../../types/database'
import { Badge } from '../ui/Badge'
import { Spinner } from '../ui/Spinner'

const PIPELINE_STAGES = [
  { key: 'not_contacted', label: 'Not Contacted', color: 'neutral' as const },
  { key: 'contacted', label: 'Contacted', color: 'info' as const },
  { key: 'replied', label: 'Replied', color: 'warning' as const },
  { key: 'partnership_active', label: 'Active', color: 'success' as const },
  { key: 'declined', label: 'Declined', color: 'danger' as const },
]

export function PipelineBoard() {
  const { data, loading, error } = useSupabase<CrmContact>({
    table: 'crm_contacts',
    order: { column: 'updated_at', ascending: false },
    limit: 500,
  })

  if (loading) return <Spinner />
  if (error) return <p className="text-sm text-[var(--color-danger)]">{error}</p>

  const byStage: Record<string, CrmContact[]> = {}
  for (const stage of PIPELINE_STAGES) {
    byStage[stage.key] = data.filter((c) => c.outreach_status === stage.key)
  }

  return (
    <div className="flex gap-4 overflow-x-auto pb-4">
      {PIPELINE_STAGES.map((stage) => {
        const contacts = byStage[stage.key] ?? []
        return (
          <div
            key={stage.key}
            className="min-w-[260px] flex-1 rounded-xl border border-[var(--color-border)] bg-[var(--color-bg)]"
          >
            {/* Column header */}
            <div className="flex items-center justify-between border-b border-[var(--color-border)] px-4 py-3">
              <div className="flex items-center gap-2">
                <Badge variant={stage.color}>{stage.label}</Badge>
              </div>
              <span className="text-xs text-[var(--color-text-muted)]">{contacts.length}</span>
            </div>

            {/* Cards */}
            <div className="max-h-[calc(100dvh-16rem)] space-y-2 overflow-y-auto p-3">
              {contacts.length === 0 ? (
                <p className="py-4 text-center text-xs text-[var(--color-text-muted)]">Empty</p>
              ) : (
                <>
                {contacts.slice(0, 30).map((contact) => {
                  const daysSince = contact.last_contacted_at
                    ? Math.floor((Date.now() - new Date(contact.last_contacted_at).getTime()) / 86400000)
                    : null

                  return (
                    <div
                      key={contact.id}
                      className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-3 transition-colors hover:bg-[var(--color-surface-hover)]"
                    >
                      <p className="text-sm font-medium">{contact.company_name}</p>
                      {contact.contact_name && (
                        <p className="text-xs text-[var(--color-text-muted)]">{contact.contact_name}</p>
                      )}
                      <div className="mt-2 flex items-center gap-2">
                        {contact.tier && (
                          <Badge variant="warning">{contact.tier.replace(/_/g, ' ')}</Badge>
                        )}
                        {daysSince != null && (
                          <span className="text-xs text-[var(--color-text-muted)]">
                            {daysSince}d ago
                          </span>
                        )}
                      </div>
                    </div>
                  )
                })}
                {contacts.length > 30 && (
                  <p className="py-2 text-center text-xs text-[var(--color-text-muted)]">
                    +{contacts.length - 30} more
                  </p>
                )}
                </>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
