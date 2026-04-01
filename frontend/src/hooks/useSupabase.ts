import { useCallback, useEffect, useRef, useState } from 'react'
import { supabase, isConfigured } from '../utils/supabase'

interface UseSupabaseOptions {
  table: string
  select?: string
  order?: { column: string; ascending?: boolean }
  limit?: number
  filters?: Record<string, string | number | boolean>
  /** Subscribe to Supabase realtime INSERT events on this table. */
  realtime?: boolean
}

interface UseSupabaseResult<T> {
  data: T[]
  loading: boolean
  error: string | null
  refetch: () => void
}

export function useSupabase<T>({
  table,
  select = '*',
  order,
  limit = 100,
  filters,
  realtime = false,
}: UseSupabaseOptions): UseSupabaseResult<T> {
  const [data, setData] = useState<T[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)

  const refetch = useCallback(() => setTick((t) => t + 1), [])

  // Keep a stable ref so the realtime callback can call the latest refetch
  const refetchRef = useRef(refetch)
  refetchRef.current = refetch

  useEffect(() => {
    if (!isConfigured || !supabase) {
      setLoading(false)
      setError(null)
      return
    }

    const client = supabase
    let cancelled = false
    setLoading(true)

    const run = async () => {
      let query = client.from(table).select(select)

      if (filters) {
        for (const [key, value] of Object.entries(filters)) {
          query = query.eq(key, value)
        }
      }
      if (order) {
        query = query.order(order.column, { ascending: order.ascending ?? false })
      }
      query = query.limit(limit)

      const { data: rows, error: err } = await query
      if (cancelled) return

      if (err) {
        setError(err.message)
        setData([])
      } else {
        setData((rows ?? []) as T[])
        setError(null)
      }
      setLoading(false)
    }

    run()
    return () => { cancelled = true }
  }, [table, select, limit, tick, JSON.stringify(order), JSON.stringify(filters)])

  // Realtime subscription — refetch on INSERT/UPDATE/DELETE
  useEffect(() => {
    if (!realtime || !isConfigured || !supabase) return

    const channel = supabase
      .channel(`${table}_changes`)
      .on(
        'postgres_changes' as any,
        { event: '*', schema: 'public', table },
        () => {
          refetchRef.current()
        },
      )
      .subscribe()

    return () => {
      supabase!.removeChannel(channel)
    }
  }, [table, realtime])

  return { data, loading, error, refetch }
}
