import { useEffect, useMemo, useState } from 'react'
import { useSupabase } from '../../hooks/useSupabase'
import { Spinner } from '../ui/Spinner'
import { KeywordStatsCards } from './KeywordStatsCards'
import { KeywordsTable } from './KeywordsTable'
import { RankMovers } from './RankMovers'
import type { RankingEntry } from '../../types/database'

export function KeywordsDashboard() {
  const { data, loading, error } = useSupabase<RankingEntry>({
    table: 'seo_our_rankings',
    order: { column: 'snapshot_date', ascending: false },
    limit: 500,
  })

  const sites = useMemo(
    () => [...new Set(data.map((r) => r.target_site))].sort(),
    [data],
  )

  const [activeSite, setActiveSite] = useState('')

  useEffect(() => {
    if (sites.length > 0 && !activeSite) {
      setActiveSite(sites[0])
    }
  }, [sites, activeSite])

  const filteredData = useMemo(
    () => (activeSite ? data.filter((r) => r.target_site === activeSite) : data),
    [data, activeSite],
  )

  if (loading) return <Spinner />

  if (error) {
    return (
      <p className="py-12 text-center text-sm text-[var(--color-danger)]">{error}</p>
    )
  }

  return (
    <div className="space-y-6">
      {/* Site filter tabs */}
      {sites.length > 1 && (
        <div className="flex gap-1 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-1">
          {sites.map((site) => (
            <button
              key={site}
              onClick={() => setActiveSite(site)}
              className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                activeSite === site
                  ? 'bg-[var(--color-primary)] text-white'
                  : 'text-[var(--color-text-muted)] hover:text-[var(--color-text)]'
              }`}
            >
              {site}
            </button>
          ))}
        </div>
      )}

      <KeywordStatsCards data={filteredData} />
      <KeywordsTable data={filteredData} />
      <RankMovers data={filteredData} />
    </div>
  )
}
