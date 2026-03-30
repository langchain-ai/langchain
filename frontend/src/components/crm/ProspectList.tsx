import { useState } from 'react'
import { useSupabase } from '../../hooks/useSupabase'
import type { BacklinkProspect } from '../../types/database'
import { Card, CardHeader, CardTitle } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Table } from '../ui/Table'
import { Spinner } from '../ui/Spinner'

const tierVariant: Record<string, 'success' | 'warning' | 'neutral'> = {
  tier_1: 'success',
  tier_2: 'warning',
  tier_3: 'neutral',
}

const columns = [
  {
    key: 'domain',
    header: 'Domain',
    render: (row: BacklinkProspect) => <span className="font-medium">{row.domain}</span>,
  },
  {
    key: 'dr',
    header: 'DR',
    render: (row: BacklinkProspect) => <span>{row.dr ?? '—'}</span>,
  },
  {
    key: 'monthly_traffic',
    header: 'Traffic',
    render: (row: BacklinkProspect) => (
      <span>{row.monthly_traffic ? row.monthly_traffic.toLocaleString() : '—'}</span>
    ),
  },
  {
    key: 'score',
    header: 'Score',
    render: (row: BacklinkProspect) => <span className="font-medium">{row.score}</span>,
  },
  {
    key: 'tier',
    header: 'Tier',
    render: (row: BacklinkProspect) => (
      row.tier ? <Badge variant={tierVariant[row.tier] ?? 'neutral'}>{row.tier.replace(/_/g, ' ')}</Badge> : <span>—</span>
    ),
  },
  {
    key: 'status',
    header: 'Status',
    render: (row: BacklinkProspect) => (
      <Badge variant={row.reply_received ? 'success' : row.status === 'new' ? 'info' : 'neutral'}>
        {row.status}
      </Badge>
    ),
  },
  {
    key: 'outreach_angle',
    header: 'Angle',
    render: (row: BacklinkProspect) => (
      <span className="max-w-xs truncate text-xs text-[var(--color-text-muted)]">
        {row.outreach_angle?.slice(0, 60) ?? '—'}
      </span>
    ),
    className: 'max-w-xs',
  },
]

export function ProspectList() {
  const [tierFilter, setTierFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState('')

  const { data, loading, error } = useSupabase<BacklinkProspect>({
    table: 'seo_backlink_prospects',
    order: { column: 'score', ascending: false },
    limit: 200,
  })

  const filtered = data.filter((p) => {
    if (tierFilter && p.tier !== tierFilter) return false
    if (statusFilter && p.status !== statusFilter) return false
    return true
  })

  const tiers = [...new Set(data.map((p) => p.tier).filter(Boolean))] as string[]
  const statuses = [...new Set(data.map((p) => p.status))]

  return (
    <Card>
      <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <CardTitle>Backlink Prospects ({filtered.length})</CardTitle>
        <div className="flex gap-2">
          <select
            value={tierFilter}
            onChange={(e) => setTierFilter(e.target.value)}
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-1.5 text-sm text-[var(--color-text)] focus:outline-none"
          >
            <option value="">All tiers</option>
            {tiers.map((t) => (
              <option key={t} value={t}>{t.replace(/_/g, ' ')}</option>
            ))}
          </select>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-1.5 text-sm text-[var(--color-text)] focus:outline-none"
          >
            <option value="">All statuses</option>
            {statuses.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
      </CardHeader>
      {loading ? (
        <Spinner />
      ) : error ? (
        <p className="text-sm text-[var(--color-danger)]">{error}</p>
      ) : (
        <Table columns={columns} data={filtered} />
      )}
    </Card>
  )
}
