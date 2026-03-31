import { useMemo, useState } from 'react'
import { ArrowUp, ArrowDown, ChevronUp, ChevronDown } from 'lucide-react'
import { Card, CardHeader, CardTitle } from '../ui/Card'
import { Badge } from '../ui/Badge'
import type { RankingEntry } from '../../types/database'

type SortKey = 'keyword' | 'position' | 'change' | 'volume' | 'url' | 'snapshot_date'

interface Props {
  data: RankingEntry[]
}

const COLUMNS: { key: SortKey; label: string }[] = [
  { key: 'keyword', label: 'Keyword' },
  { key: 'position', label: 'Position' },
  { key: 'change', label: 'Change' },
  { key: 'volume', label: 'Volume' },
  { key: 'url', label: 'URL' },
  { key: 'snapshot_date', label: 'Date' },
]

function formatUrl(url: string): string {
  if (!url) return '—'
  try {
    return new URL(url).pathname
  } catch {
    return url
  }
}

export function KeywordsTable({ data }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>('position')
  const [sortAsc, setSortAsc] = useState(true)

  const sorted = useMemo(() => {
    const copy = [...data]
    copy.sort((a, b) => {
      let aVal: string | number = a[sortKey] ?? 0
      let bVal: string | number = b[sortKey] ?? 0
      if (typeof aVal === 'string') aVal = aVal.toLowerCase()
      if (typeof bVal === 'string') bVal = bVal.toLowerCase()
      if (aVal < bVal) return sortAsc ? -1 : 1
      if (aVal > bVal) return sortAsc ? 1 : -1
      return 0
    })
    return copy
  }, [data, sortKey, sortAsc])

  const handleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortAsc((prev) => !prev)
    } else {
      setSortKey(key)
      setSortAsc(true)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Rankings ({data.length})</CardTitle>
      </CardHeader>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--color-border)] text-left text-xs text-[var(--color-text-muted)]">
              {COLUMNS.map((col) => {
                const isUrl = col.label === 'URL'
                return (
                  <th
                    key={col.label}
                    className={`px-3 py-2.5 font-medium ${isUrl ? '' : 'cursor-pointer select-none hover:text-[var(--color-text)]'}`}
                    onClick={isUrl ? undefined : () => handleSort(col.key)}
                  >
                    <span className="inline-flex items-center gap-1">
                      {col.label}
                      {!isUrl && sortKey === col.key && (
                        sortAsc
                          ? <ChevronUp size={14} />
                          : <ChevronDown size={14} />
                      )}
                    </span>
                  </th>
                )
              })}
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 ? (
              <tr>
                <td
                  colSpan={COLUMNS.length}
                  className="px-3 py-8 text-center text-[var(--color-text-muted)]"
                >
                  No ranking data available
                </td>
              </tr>
            ) : (
              sorted.map((row) => (
                <tr
                  key={row.id}
                  className="border-b border-[var(--color-border)] last:border-0 hover:bg-[var(--color-surface-hover)] transition-colors"
                >
                  <td className="px-3 py-2.5 font-medium">{row.keyword}</td>
                  <td className="px-3 py-2.5">{row.position ?? '—'}</td>
                  <td className="px-3 py-2.5">
                    <ChangeCell change={row.change} />
                  </td>
                  <td className="px-3 py-2.5">{(row.volume ?? 0).toLocaleString()}</td>
                  <td className="px-3 py-2.5 max-w-[200px] truncate text-xs text-[var(--color-text-muted)]">
                    {formatUrl(row.url)}
                  </td>
                  <td className="px-3 py-2.5 text-[var(--color-text-muted)]">
                    {row.snapshot_date
                      ? new Date(row.snapshot_date).toLocaleDateString()
                      : '—'}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function ChangeCell({ change }: { change: number }) {
  if (change > 0) {
    return (
      <Badge variant="success">
        <ArrowUp size={12} className="mr-0.5" />+{change}
      </Badge>
    )
  }
  if (change < 0) {
    return (
      <Badge variant="danger">
        <ArrowDown size={12} className="mr-0.5" />{change}
      </Badge>
    )
  }
  return <Badge variant="neutral">—</Badge>
}
