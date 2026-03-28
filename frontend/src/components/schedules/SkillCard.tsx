import { Loader2 } from 'lucide-react'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import {
  FREQUENCY_OPTIONS,
  CATEGORY_VARIANTS,
  type FrequencyOption,
  type SkillMeta,
} from '../../data/skills'

interface Props {
  skill: SkillMeta
  frequency: FrequencyOption
  onFrequencyChange: (freq: FrequencyOption) => void
  saving: boolean
}

function formatName(name: string): string {
  return name
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ')
}

const costColors: Record<string, string> = {
  sonnet: 'warning',
  haiku: 'info',
  opus: 'danger',
  none: 'neutral',
}

export function SkillCard({ skill, frequency, onFrequencyChange, saving }: Props) {
  const categoryVariant = CATEGORY_VARIANTS[skill.category] as
    | 'success'
    | 'warning'
    | 'danger'
    | 'info'
    | 'neutral'

  return (
    <Card className="relative">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-semibold text-[var(--color-text)]">
              {formatName(skill.name)}
            </h4>
            <Badge variant={categoryVariant}>{skill.category}</Badge>
            <Badge variant={costColors[skill.costTier] as any}>{skill.costTier}</Badge>
          </div>
          <p className="mt-1 text-xs text-[var(--color-text-muted)]">{skill.description}</p>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          {saving && <Loader2 size={14} className="animate-spin text-[var(--color-text-muted)]" />}
          <select
            value={frequency}
            onChange={(e) => onFrequencyChange(e.target.value as FrequencyOption)}
            disabled={saving}
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-2.5 py-1.5 text-xs font-medium text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-blue-500/40 disabled:opacity-50"
          >
            {FREQUENCY_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      </div>
    </Card>
  )
}
