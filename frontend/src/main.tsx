import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { isConfigured } from './utils/supabase'

if (!isConfigured) {
  console.warn('[AgentHQ] VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY not set — data features disabled.')
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
