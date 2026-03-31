import { X, Mail, Phone, Globe, MapPin } from 'lucide-react'
import { useSupabase } from '../../hooks/useSupabase'
import type { CrmContact, CrmInteraction } from '../../types/database'
import { Badge } from '../ui/Badge'
import { Spinner } from '../ui/Spinner'

interface Props {
  contact: CrmContact
  onClose: () => void
}

export function ContactDetail({ contact, onClose }: Props) {
  const { data: interactions, loading: loadingInteractions, error: interactionsError } = useSupabase<CrmInteraction>({
    table: 'crm_interactions',
    filters: { contact_id: contact.id },
    order: { column: 'created_at', ascending: false },
    limit: 50,
  })

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/40" onClick={onClose}>
      <div
        className="h-full w-full max-w-lg overflow-y-auto border-l border-[var(--color-border)] bg-[var(--color-surface)] p-6"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-lg font-semibold">{contact.company_name}</h2>
            {contact.contact_name && (
              <p className="text-sm text-[var(--color-text-muted)]">
                {contact.contact_name} {contact.contact_role ? `· ${contact.contact_role}` : ''}
              </p>
            )}
          </div>
          <button onClick={onClose} className="rounded p-1 hover:bg-[var(--color-surface-hover)]">
            <X size={18} />
          </button>
        </div>

        {/* Details */}
        <div className="mt-4 space-y-2 text-sm">
          {contact.email && (
            <div className="flex items-center gap-2 text-[var(--color-text-muted)]">
              <Mail size={14} /> {contact.email}
            </div>
          )}
          {contact.phone && (
            <div className="flex items-center gap-2 text-[var(--color-text-muted)]">
              <Phone size={14} /> {contact.phone}
            </div>
          )}
          {contact.website && (
            <div className="flex items-center gap-2 text-[var(--color-text-muted)]">
              <Globe size={14} /> {contact.website}
            </div>
          )}
          {(contact.city || contact.region) && (
            <div className="flex items-center gap-2 text-[var(--color-text-muted)]">
              <MapPin size={14} /> {[contact.city, contact.region, contact.postcode].filter(Boolean).join(', ')}
            </div>
          )}
        </div>

        {/* Tags */}
        <div className="mt-4 flex flex-wrap gap-2">
          <Badge variant="info">{contact.category.replace(/_/g, ' ')}</Badge>
          {contact.tier && <Badge variant="warning">{contact.tier.replace(/_/g, ' ')}</Badge>}
          <Badge variant={contact.outreach_status === 'partnership_active' ? 'success' : 'neutral'}>
            {contact.outreach_status.replace(/_/g, ' ')}
          </Badge>
          <Badge variant="neutral">Score: {contact.score}</Badge>
        </div>

        {contact.notes && (
          <div className="mt-4 rounded-lg bg-[var(--color-bg)] p-3 text-sm text-[var(--color-text-muted)]">
            {contact.notes}
          </div>
        )}

        {/* Interaction timeline */}
        <h3 className="mb-3 mt-6 text-sm font-semibold">
          Interactions {loadingInteractions ? '' : `(${interactions.length})`}
        </h3>
        {loadingInteractions ? (
          <Spinner />
        ) : interactionsError ? (
          <p className="text-sm text-[var(--color-danger)]">{interactionsError}</p>
        ) : interactions.length === 0 ? (
          <p className="text-sm text-[var(--color-text-muted)]">No interactions recorded</p>
        ) : (
          <div className="space-y-3">
            {interactions.map((i) => (
              <div
                key={i.id}
                className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] p-3"
              >
                <div className="flex items-center gap-2">
                  <Badge variant="info">{i.interaction_type.replace(/_/g, ' ')}</Badge>
                  <span className="text-xs text-[var(--color-text-muted)]">{i.direction}</span>
                  <span className="ml-auto text-xs text-[var(--color-text-muted)]">
                    {new Date(i.created_at).toLocaleString()}
                  </span>
                </div>
                {i.subject && <p className="mt-1 text-sm font-medium">{i.subject}</p>}
                {i.body_preview && (
                  <p className="mt-1 text-xs text-[var(--color-text-muted)]">{i.body_preview}</p>
                )}
                <p className="mt-1 text-xs text-[var(--color-text-muted)]">by {i.performed_by}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
