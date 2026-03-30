import { Mail, Phone, MessageSquare, FileText, Send } from 'lucide-react'
import { useSupabase } from '../../hooks/useSupabase'
import type { CrmInteraction } from '../../types/database'
import { Card, CardHeader, CardTitle } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Spinner } from '../ui/Spinner'

const typeIcons: Record<string, React.ReactNode> = {
  email_sent: <Send size={14} className="text-blue-400" />,
  email_received: <Mail size={14} className="text-emerald-400" />,
  phone_call: <Phone size={14} className="text-amber-400" />,
  meeting: <MessageSquare size={14} className="text-purple-400" />,
  note: <FileText size={14} className="text-slate-400" />,
  social_dm: <MessageSquare size={14} className="text-pink-400" />,
}

export function InteractionLog() {
  const { data, loading, error } = useSupabase<CrmInteraction>({
    table: 'crm_interactions',
    order: { column: 'created_at', ascending: false },
    limit: 50,
  })

  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Activity</CardTitle>
      </CardHeader>
      {loading ? (
        <Spinner />
      ) : error ? (
        <p className="text-sm text-[var(--color-danger)]">{error}</p>
      ) : data.length === 0 ? (
        <p className="py-8 text-center text-sm text-[var(--color-text-muted)]">No interactions yet</p>
      ) : (
        <div className="space-y-3">
          {data.map((interaction) => (
            <div
              key={interaction.id}
              className="flex gap-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] p-3"
            >
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[var(--color-surface)]">
                {typeIcons[interaction.interaction_type] ?? <Mail size={14} />}
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <Badge variant="info">
                    {interaction.interaction_type.replace(/_/g, ' ')}
                  </Badge>
                  <span className="text-xs text-[var(--color-text-muted)]">{interaction.direction}</span>
                  <span className="ml-auto text-xs text-[var(--color-text-muted)]">
                    {new Date(interaction.created_at).toLocaleString()}
                  </span>
                </div>
                {interaction.subject && (
                  <p className="mt-1 text-sm font-medium">{interaction.subject}</p>
                )}
                {interaction.body_preview && (
                  <p className="mt-0.5 text-xs text-[var(--color-text-muted)]">
                    {interaction.body_preview.slice(0, 150)}
                  </p>
                )}
                <p className="mt-1 text-xs text-[var(--color-text-muted)]">by {interaction.performed_by}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}
