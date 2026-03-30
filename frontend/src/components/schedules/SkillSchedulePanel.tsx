import { useScheduleEntries } from '../../hooks/useScheduleEntries'
import {
  SKILLS,
  CATEGORY_ORDER,
  CATEGORY_LABELS,
  type FrequencyOption,
} from '../../data/skills'
import { Card, CardHeader, CardTitle } from '../ui/Card'
import { Spinner } from '../ui/Spinner'
import { SkillCard } from './SkillCard'

export function SkillSchedulePanel() {
  const { loading, error, deriveFrequency, updateFrequency, savingSkill } = useScheduleEntries()

  if (loading) {
    return (
      <Card>
        <div className="flex items-center justify-center py-8">
          <Spinner />
        </div>
      </Card>
    )
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Skill Schedule</CardTitle>
        </CardHeader>
        <p className="text-sm text-[var(--color-danger)]">{error}</p>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Skill Schedule</CardTitle>
      </CardHeader>
      <div className="space-y-6">
        {CATEGORY_ORDER.map((cat) => {
          const catSkills = SKILLS.filter((s) => s.category === cat)
          if (catSkills.length === 0) return null

          return (
            <section key={cat}>
              <h4 className="mb-3 text-xs font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
                {CATEGORY_LABELS[cat]}
              </h4>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                {catSkills.map((skill) => (
                  <SkillCard
                    key={skill.name}
                    skill={skill}
                    frequency={deriveFrequency(skill.name)}
                    onFrequencyChange={(freq: FrequencyOption) =>
                      updateFrequency(skill.name, freq, skill.category)
                    }
                    saving={savingSkill === skill.name}
                  />
                ))}
              </div>
            </section>
          )
        })}
      </div>
    </Card>
  )
}
