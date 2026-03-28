import { useState } from 'react'
import { ContactsTable } from './ContactsTable'
import { PipelineBoard } from './PipelineBoard'
import { ProspectList } from './ProspectList'
import { InteractionLog } from './InteractionLog'

const TABS = [
  { key: 'contacts', label: 'Contacts' },
  { key: 'pipeline', label: 'Pipeline' },
  { key: 'prospects', label: 'Prospects' },
  { key: 'activity', label: 'Activity' },
] as const

type TabKey = typeof TABS[number]['key']

export function CrmDashboard() {
  const [activeTab, setActiveTab] = useState<TabKey>('contacts')

  return (
    <div className="space-y-6">
      {/* Tab bar */}
      <div className="flex gap-1 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-1">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === tab.key
                ? 'bg-[var(--color-primary)] text-white'
                : 'text-[var(--color-text-muted)] hover:text-[var(--color-text)]'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'contacts' && <ContactsTable />}
      {activeTab === 'pipeline' && <PipelineBoard />}
      {activeTab === 'prospects' && <ProspectList />}
      {activeTab === 'activity' && <InteractionLog />}
    </div>
  )
}
