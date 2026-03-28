import { useCallback, useState } from 'react'
import { supabase, isConfigured } from '../utils/supabase'
import { useSupabase } from './useSupabase'
import type { ScheduleEntry } from '../types/database'
import type { FrequencyOption } from '../data/skills'

/** Default day assignments by category when creating new schedule rows. */
const DEFAULT_DAYS: Record<string, number[]> = {
  content: [0, 1, 2],       // Mon, Tue, Wed
  prospecting: [3, 3, 4],   // Thu, Thu, Fri
  analytics: [4, 1, 3],     // Fri, Tue, Thu
  maintenance: [5, 6, 5],   // Sat, Sun, Sat
}

function getDaysForSkill(category: string, count: number): number[] {
  const days = DEFAULT_DAYS[category] ?? [0, 2, 4]
  return days.slice(0, count)
}

export function useScheduleEntries() {
  const { data: entries, loading, error, refetch } = useSupabase<ScheduleEntry>({
    table: 'ralf_schedule',
    limit: 200,
  })

  const [savingSkill, setSavingSkill] = useState<string | null>(null)

  const deriveFrequency = useCallback(
    (skillName: string): FrequencyOption => {
      const rows = entries.filter((e) => e.skill === skillName && e.active)

      const dailyCount = rows.filter((r) => r.cadence === 'daily').length
      const weeklyCount = rows.filter((r) => r.cadence === 'weekly').length
      const monthlyCount = rows.filter((r) => r.cadence === 'monthly').length

      if (dailyCount >= 3) return '3x_week'
      if (dailyCount >= 2) return '2x_week'
      if (dailyCount === 1 || weeklyCount >= 1) return 'weekly'
      if (monthlyCount >= 1) return 'monthly'

      // Check if rows exist but are all inactive
      const allRows = entries.filter((e) => e.skill === skillName)
      if (allRows.length > 0) return 'off'

      // No rows at all — treat as weekly (default)
      return 'weekly'
    },
    [entries],
  )

  const updateFrequency = useCallback(
    async (skillName: string, frequency: FrequencyOption, category: string) => {
      if (!isConfigured) return
      setSavingSkill(skillName)

      try {
        // Deactivate all existing rows for this skill
        await supabase
          .from('ralf_schedule')
          .update({ active: false, updated_at: new Date().toISOString() })
          .eq('skill', skillName)

        if (frequency === 'off') {
          refetch()
          return
        }

        // Build new rows based on frequency
        const newRows: Partial<ScheduleEntry>[] = []
        const boost = 30

        if (frequency === '3x_week') {
          const days = getDaysForSkill(category, 3)
          for (const day of days) {
            newRows.push({
              cadence: 'daily',
              day_of_week: day,
              skill: skillName,
              boost_amount: boost,
              label: `${skillName} (3x/week)`,
              description: `Scheduled 3x per week`,
              active: true,
            })
          }
        } else if (frequency === '2x_week') {
          const days = getDaysForSkill(category, 2)
          for (const day of days) {
            newRows.push({
              cadence: 'daily',
              day_of_week: day,
              skill: skillName,
              boost_amount: boost,
              label: `${skillName} (2x/week)`,
              description: `Scheduled 2x per week`,
              active: true,
            })
          }
        } else if (frequency === 'weekly') {
          const days = getDaysForSkill(category, 1)
          newRows.push({
            cadence: 'weekly',
            day_of_week: days[0],
            skill: skillName,
            boost_amount: boost,
            label: `${skillName} (weekly)`,
            description: `Scheduled weekly`,
            active: true,
          })
        } else if (frequency === 'fortnightly') {
          const days = getDaysForSkill(category, 1)
          newRows.push({
            cadence: 'weekly',
            day_of_week: days[0],
            skill: skillName,
            boost_amount: 20,
            label: `${skillName} (fortnightly)`,
            description: `Scheduled fortnightly`,
            active: true,
          })
        } else if (frequency === 'monthly') {
          newRows.push({
            cadence: 'monthly',
            day_of_month: 1,
            skill: skillName,
            boost_amount: 40,
            label: `${skillName} (monthly)`,
            description: `Scheduled monthly`,
            active: true,
          })
        }

        if (newRows.length > 0) {
          await supabase.from('ralf_schedule').insert(newRows)
        }

        refetch()
      } finally {
        setSavingSkill(null)
      }
    },
    [refetch],
  )

  return { entries, loading, error, deriveFrequency, updateFrequency, savingSkill }
}
