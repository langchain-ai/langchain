import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Shell } from './components/layout/Shell'
import { AgentDashboard } from './components/agents/AgentDashboard'
import { ScheduleDashboard } from './components/schedules/ScheduleDashboard'
import { OrgChart } from './components/orgchart/OrgChart'
import { CrmDashboard } from './components/crm/CrmDashboard'
import { KeywordsDashboard } from './components/keywords/KeywordsDashboard'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Shell />}>
          <Route path="/agents" element={<AgentDashboard />} />
          <Route path="/schedules" element={<ScheduleDashboard />} />
          <Route path="/org-chart" element={<OrgChart />} />
          <Route path="/crm" element={<CrmDashboard />} />
          <Route path="/keywords" element={<KeywordsDashboard />} />
          <Route path="*" element={<Navigate to="/agents" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
