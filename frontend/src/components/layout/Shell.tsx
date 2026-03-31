import { useState } from 'react'
import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { Header } from './Header'

export function Shell() {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="h-screen overflow-hidden">
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <div className="flex h-full flex-col md:ml-56">
        <Header onMenuToggle={() => setSidebarOpen(true)} />
        <main className="min-h-0 flex-1 overflow-y-auto p-4 md:p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
