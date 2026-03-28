import { useCallback, useMemo } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  Position,
  Handle,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

/* ------------------------------------------------------------------ */
/*  Custom node                                                        */
/* ------------------------------------------------------------------ */
interface NodeData {
  label: string
  role: string
  color: string
  [key: string]: unknown
}

function OrgNode({ data }: { data: NodeData }) {
  return (
    <div
      className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3 text-center shadow-lg"
      style={{ minWidth: 140 }}
    >
      <Handle type="target" position={Position.Top} className="!bg-[var(--color-border)]" />
      <div
        className="mx-auto mb-2 flex h-8 w-8 items-center justify-center rounded-lg text-sm font-bold text-white"
        style={{ background: data.color }}
      >
        {data.label.charAt(0)}
      </div>
      <p className="text-sm font-semibold text-[var(--color-text)]">{data.label}</p>
      <p className="text-xs text-[var(--color-text-muted)]">{data.role}</p>
      <Handle type="source" position={Position.Bottom} className="!bg-[var(--color-border)]" />
    </div>
  )
}

const nodeTypes = { org: OrgNode }

/* ------------------------------------------------------------------ */
/*  Tree data                                                          */
/* ------------------------------------------------------------------ */
const NODES: Node<NodeData>[] = [
  // Level 0 — Owner
  { id: 'owner', position: { x: 400, y: 0 }, data: { label: 'Owner', role: 'Human Operator', color: '#6366f1' }, type: 'org' },

  // Level 1 — Agents
  { id: 'ralf', position: { x: 200, y: 120 }, data: { label: 'Ralf', role: 'SEO Agent', color: '#3b82f6' }, type: 'org' },
  { id: 'scraper', position: { x: 600, y: 120 }, data: { label: 'Scraper', role: 'Data Agent', color: '#8b5cf6' }, type: 'org' },

  // Level 2 — Ralf sub-systems
  { id: 'worker', position: { x: 0, y: 260 }, data: { label: 'Worker', role: 'Every 3h', color: '#22c55e' }, type: 'org' },
  { id: 'pulse', position: { x: 200, y: 260 }, data: { label: 'Pulse', role: 'Every 60min', color: '#f59e0b' }, type: 'org' },
  { id: 'heartbeat', position: { x: 400, y: 260 }, data: { label: 'Heartbeat', role: 'Orchestrator', color: '#ef4444' }, type: 'org' },

  // Level 3 — Worker nodes
  { id: 'content', position: { x: -120, y: 400 }, data: { label: 'Content', role: 'Writer + Briefs + Gaps', color: '#06b6d4' }, type: 'org' },
  { id: 'outreach', position: { x: 80, y: 400 }, data: { label: 'Outreach', role: 'Prospect + Email + Score', color: '#f97316' }, type: 'org' },
  { id: 'keywords', position: { x: -120, y: 530 }, data: { label: 'Keywords', role: 'Research + Cache', color: '#a855f7' }, type: 'org' },
  { id: 'linking', position: { x: 80, y: 530 }, data: { label: 'Linking', role: 'Internal Linker', color: '#ec4899' }, type: 'org' },

  // Level 3 — Pulse nodes
  { id: 'rankings', position: { x: 200, y: 400 }, data: { label: 'Rankings', role: 'Rank Tracker + Movers', color: '#14b8a6' }, type: 'org' },
  { id: 'reporting', position: { x: 350, y: 400 }, data: { label: 'Reporting', role: 'Progress Summaries', color: '#64748b' }, type: 'org' },

  // Scraper child
  { id: 'web-scrape', position: { x: 600, y: 260 }, data: { label: 'Scraper', role: 'Web Data Collection', color: '#8b5cf6' }, type: 'org' },
]

const EDGES: Edge[] = [
  { id: 'e-owner-ralf', source: 'owner', target: 'ralf' },
  { id: 'e-owner-scraper', source: 'owner', target: 'scraper' },
  { id: 'e-ralf-worker', source: 'ralf', target: 'worker' },
  { id: 'e-ralf-pulse', source: 'ralf', target: 'pulse' },
  { id: 'e-ralf-heartbeat', source: 'ralf', target: 'heartbeat' },
  { id: 'e-worker-content', source: 'worker', target: 'content' },
  { id: 'e-worker-outreach', source: 'worker', target: 'outreach' },
  { id: 'e-worker-keywords', source: 'worker', target: 'keywords' },
  { id: 'e-worker-linking', source: 'worker', target: 'linking' },
  { id: 'e-pulse-rankings', source: 'pulse', target: 'rankings' },
  { id: 'e-pulse-reporting', source: 'pulse', target: 'reporting' },
  { id: 'e-scraper-web', source: 'scraper', target: 'web-scrape' },
]

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */
export function OrgChart() {
  const styledEdges = useMemo(
    () =>
      EDGES.map((e) => ({
        ...e,
        style: { stroke: '#475569', strokeWidth: 2 },
        animated: false,
      })),
    [],
  )

  const onInit = useCallback(() => {}, [])

  return (
    <div className="h-[calc(100vh-8rem)] w-full rounded-xl border border-[var(--color-border)] bg-[var(--color-bg)]">
      <ReactFlow
        nodes={NODES}
        edges={styledEdges}
        nodeTypes={nodeTypes}
        onInit={onInit}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        proOptions={{ hideAttribution: true }}
        minZoom={0.3}
        maxZoom={1.5}
      >
        <Background color="#1e293b" gap={20} />
        <Controls
          className="!border-[var(--color-border)] !bg-[var(--color-surface)] [&>button]:!border-[var(--color-border)] [&>button]:!bg-[var(--color-surface)] [&>button]:!fill-[var(--color-text-muted)]"
        />
      </ReactFlow>
    </div>
  )
}
