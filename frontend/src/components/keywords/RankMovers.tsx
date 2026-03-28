import { useMemo } from 'react'
import { ArrowUp, ArrowDown } from 'lucide-react'
import { Card, CardHeader, CardTitle } from '../ui/Card'
import { Badge } from '../ui/Badge'
import type { RankingEntry } from '../../types/database'

interface Props {
  data: RankingEntry[]
}

export function RankMovers({ data }: Props) {
  const { winners, losers } = useMemo(() => {
    const w = data
      .filter((r) => r.change > 0)
      .sort((a, b) => b.change - a.change)
      .slice(0, 5)

    const l = data
      .filter((r) => r.change < 0)
      .sort((a, b) => a.change - b.change)
      .slice(0, 5)

    return { winners: w, losers: l }
  }, [data])

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Winners</CardTitle>
        </CardHeader>
        {winners.length === 0 ? (
          <p className="text-sm text-[var(--color-text-muted)]">No position gains</p>
        ) : (
          <ul className="space-y-3">
            {winners.map((row) => (
              <MoverRow key={row.id} entry={row} direction="up" />
            ))}
          </ul>
        )}
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Losers</CardTitle>
        </CardHeader>
        {losers.length === 0 ? (
          <p className="text-sm text-[var(--color-text-muted)]">No position drops</p>
        ) : (
          <ul className="space-y-3">
            {losers.map((row) => (
              <MoverRow key={row.id} entry={row} direction="down" />
            ))}
          </ul>
        )}
      </Card>
    </div>
  )
}

function MoverRow({
  entry,
  direction,
}: {
  entry: RankingEntry
  direction: 'up' | 'down'
}) {
  const prev = entry.position - entry.change
  const Icon = direction === 'up' ? ArrowUp : ArrowDown
  const variant = direction === 'up' ? 'success' : 'danger'
  const sign = direction === 'up' ? '+' : ''

  return (
    <li className="flex items-center justify-between">
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium">{entry.keyword}</p>
        <p className="text-xs text-[var(--color-text-muted)]">
          {prev} → {entry.position}
        </p>
      </div>
      <Badge variant={variant}>
        <Icon size={12} className="mr-0.5" />{sign}{entry.change}
      </Badge>
    </li>
  )
}
