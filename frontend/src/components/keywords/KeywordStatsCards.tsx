import { useMemo } from 'react'
import { Hash, Target, Trophy, TrendingUp } from 'lucide-react'
import { Card } from '../ui/Card'
import type { RankingEntry } from '../../types/database'

interface Props {
  data: RankingEntry[]
}

export function KeywordStatsCards({ data }: Props) {
  const stats = useMemo(() => {
    const uniqueKeywords = new Set(data.map((r) => r.keyword)).size
    const withPosition = data.filter((r) => r.position != null && r.position > 0)
    const avgPosition =
      withPosition.length > 0
        ? (withPosition.reduce((sum, r) => sum + r.position, 0) / withPosition.length).toFixed(1)
        : '—'
    const top10 = data.filter((r) => r.position > 0 && r.position <= 10).length

    let biggestMoverLabel = '—'
    if (data.length > 0) {
      const sorted = [...data].sort(
        (a, b) => Math.abs(b.change ?? 0) - Math.abs(a.change ?? 0),
      )
      const top = sorted[0]
      if (top && top.change !== 0) {
        const sign = top.change > 0 ? '+' : ''
        biggestMoverLabel = `${top.keyword} (${sign}${top.change})`
      }
    }

    return [
      { label: 'Total Keywords', value: uniqueKeywords, icon: Hash },
      { label: 'Avg Position', value: avgPosition, icon: Target },
      { label: 'Top 10', value: top10, icon: Trophy },
      { label: 'Biggest Mover', value: biggestMoverLabel, icon: TrendingUp },
    ]
  }, [data])

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {stats.map((stat) => {
        const Icon = stat.icon
        return (
          <Card key={stat.label}>
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--color-primary)]/10">
                <Icon size={18} className="text-[var(--color-primary)]" />
              </div>
              <div>
                <p className="text-xs text-[var(--color-text-muted)]">{stat.label}</p>
                <p className="mt-0.5 text-xl font-semibold truncate max-w-[180px]">
                  {stat.value}
                </p>
              </div>
            </div>
          </Card>
        )
      })}
    </div>
  )
}
