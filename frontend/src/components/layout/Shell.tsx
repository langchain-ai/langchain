import { useCallback, useState } from 'react'
import { Outlet, useOutletContext } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { Header } from './Header'

export type ShellContext = { registerRefetch: (fn: () => void) => void }

export function Shell() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [refetchFns, setRefetchFns] = useState<Array<() => void>>([])

  const registerRefetch = useCallback((fn: () => void) => {
    setRefetchFns((prev) => (prev.includes(fn) ? prev : [...prev, fn]))
    const unregister = () => setRefetchFns((prev) => prev.filter((f) => f !== fn))
    return unregister
  }, [])

  const handleRefresh = useCallback(() => {
    for (const fn of refetchFns) fn()
  }, [refetchFns])

  return (
    <div className="h-screen overflow-hidden">
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <div className="flex h-full flex-col md:ml-56">
        <Header onMenuToggle={() => setSidebarOpen(true)} onRefresh={handleRefresh} />
        <main className="min-h-0 flex-1 overflow-y-auto p-4 md:p-6">
          <Outlet context={{ registerRefetch } satisfies ShellContext} />
        </main>
      </div>
    </div>
  )
}

export function useShellContext() {
  return useOutletContext<ShellContext>()
}
