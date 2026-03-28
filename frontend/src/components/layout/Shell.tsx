import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { Header } from './Header'

export function Shell() {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="ml-56 flex flex-1 flex-col">
        <Header />
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
