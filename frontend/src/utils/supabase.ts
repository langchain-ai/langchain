import { createClient, type SupabaseClient } from '@supabase/supabase-js'

const url = import.meta.env.VITE_SUPABASE_URL ?? ''
const key = import.meta.env.VITE_SUPABASE_ANON_KEY ?? ''

export const isConfigured = Boolean(url && key)

// Only create a real client when env vars are present
export const supabase: SupabaseClient | null = isConfigured
  ? createClient(url, key)
  : null
