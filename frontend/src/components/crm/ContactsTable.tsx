import { useState } from 'react'
import { useSupabase } from '../../hooks/useSupabase'
import { isConfigured } from '../../utils/supabase'
import type { CrmContact } from '../../types/database'
import { Card, CardHeader, CardTitle } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Table } from '../ui/Table'
import { Spinner } from '../ui/Spinner'
import { ContactDetail } from './ContactDetail'

const statusVariant: Record<string, 'success' | 'warning' | 'danger' | 'info' | 'neutral'> = {
  not_contacted: 'neutral',
  contacted: 'info',
  replied: 'warning',
  partnership_active: 'success',
  declined: 'danger',
  blocked: 'danger',
}

const columns = [
  {
    key: 'company_name',
    header: 'Company',
    render: (row: CrmContact) => <span className="font-medium">{row.company_name}</span>,
  },
  {
    key: 'contact_name',
    header: 'Contact',
    render: (row: CrmContact) => <span>{row.contact_name ?? '—'}</span>,
  },
  {
    key: 'email',
    header: 'Email',
    render: (row: CrmContact) => (
      <span className="text-xs text-[var(--color-text-muted)]">{row.email ?? '—'}</span>
    ),
  },
  {
    key: 'category',
    header: 'Category',
    render: (row: CrmContact) => (
      <Badge variant="info">{row.category.replace(/_/g, ' ')}</Badge>
    ),
  },
  {
    key: 'outreach_status',
    header: 'Status',
    render: (row: CrmContact) => (
      <Badge variant={statusVariant[row.outreach_status] ?? 'neutral'}>
        {row.outreach_status.replace(/_/g, ' ')}
      </Badge>
    ),
  },
  {
    key: 'tier',
    header: 'Tier',
    render: (row: CrmContact) => (
      <span className="text-xs">{row.tier?.replace(/_/g, ' ') ?? '—'}</span>
    ),
  },
  {
    key: 'score',
    header: 'Score',
    render: (row: CrmContact) => <span className="text-sm font-medium">{row.score}</span>,
  },
]

export function ContactsTable() {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [categoryFilter, setCategoryFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [search, setSearch] = useState('')

  const { data, loading, error } = useSupabase<CrmContact>({
    table: 'crm_contacts',
    order: { column: 'updated_at', ascending: false },
    limit: 200,
  })

  const filtered = data.filter((c) => {
    if (categoryFilter && c.category !== categoryFilter) return false
    if (statusFilter && c.outreach_status !== statusFilter) return false
    if (search) {
      const q = search.toLowerCase()
      return (
        c.company_name.toLowerCase().includes(q) ||
        (c.contact_name?.toLowerCase().includes(q) ?? false) ||
        (c.email?.toLowerCase().includes(q) ?? false)
      )
    }
    return true
  })

  const categories = [...new Set(data.map((c) => c.category))]
  const statuses = [...new Set(data.map((c) => c.outreach_status))]

  const selected = data.find((c) => c.id === selectedId) ?? null

  return (
    <>
      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <CardTitle>Contacts ({filtered.length})</CardTitle>
          <div className="flex flex-wrap gap-2">
            <input
              type="text"
              placeholder="Search..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-1.5 text-sm text-[var(--color-text)] placeholder:text-[var(--color-text-muted)] focus:outline-none focus:ring-1 focus:ring-[var(--color-primary)]"
            />
            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-1.5 text-sm text-[var(--color-text)] focus:outline-none"
            >
              <option value="">All categories</option>
              {categories.map((c) => (
                <option key={c} value={c}>{c.replace(/_/g, ' ')}</option>
              ))}
            </select>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-1.5 text-sm text-[var(--color-text)] focus:outline-none"
            >
              <option value="">All statuses</option>
              {statuses.map((s) => (
                <option key={s} value={s}>{s.replace(/_/g, ' ')}</option>
              ))}
            </select>
          </div>
        </CardHeader>
        {!isConfigured ? (
          <p className="p-4 text-sm text-[var(--color-warning)]">
            Supabase is not configured. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to enable data.
          </p>
        ) : loading ? (
          <Spinner />
        ) : error ? (
          <p className="p-4 text-sm text-[var(--color-danger)]">{error}</p>
        ) : filtered.length === 0 ? (
          <p className="p-4 text-sm text-[var(--color-text-muted)]">
            No contacts found. If data was expected, verify RLS policies allow SELECT for the anon role.
          </p>
        ) : (
          <Table
            columns={columns}
            data={filtered}
            onRowClick={(row) => setSelectedId(row.id as string)}
          />
        )}
      </Card>

      {selected && <ContactDetail contact={selected} onClose={() => setSelectedId(null)} />}
    </>
  )
}
