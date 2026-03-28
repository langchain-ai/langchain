import { useMemo } from 'react'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { useSupabase } from '../../hooks/useSupabase'
import type { LlmCostLog } from '../../types/database'
import { Card, CardHeader, CardTitle } from '../ui/Card'
import { Spinner } from '../ui/Spinner'

export function CostChart() {
  const { data, loading, error } = useSupabase<LlmCostLog>({
    table: 'llm_cost_log',
    order: { column: 'created_at', ascending: true },
    limit: 500,
  })

  const chartData = useMemo(() => {
    const byDay: Record<string, number> = {}
    for (const row of data) {
      const day = row.created_at.slice(0, 10)
      byDay[day] = (byDay[day] ?? 0) + row.cost_usd
    }
    let cumulative = 0
    return Object.entries(byDay).map(([date, cost]) => {
      cumulative += cost
      return { date, daily: +cost.toFixed(4), cumulative: +cumulative.toFixed(4) }
    })
  }, [data])

  const totalSpend = chartData.length > 0 ? chartData[chartData.length - 1].cumulative : 0

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>LLM Cost</CardTitle>
        <span className="text-sm font-medium text-[var(--color-text-muted)]">
          Total: ${totalSpend.toFixed(2)}
        </span>
      </CardHeader>
      {loading ? (
        <Spinner />
      ) : error ? (
        <p className="text-sm text-[var(--color-text-muted)]">{error}</p>
      ) : chartData.length === 0 ? (
        <p className="py-8 text-center text-sm text-[var(--color-text-muted)]">No cost data yet</p>
      ) : (
        <ResponsiveContainer width="100%" height={240}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="costGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 11 }} tickLine={false} />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} tickLine={false} tickFormatter={(v) => `$${v}`} />
            <Tooltip
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: '#f1f5f9' }}
              formatter={(value: number, name: string) => [`$${value.toFixed(4)}`, name === 'daily' ? 'Daily' : 'Cumulative']}
            />
            <Area type="monotone" dataKey="daily" stroke="#3b82f6" fill="url(#costGrad)" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </Card>
  )
}
