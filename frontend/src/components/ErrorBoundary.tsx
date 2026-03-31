import { Component, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  render() {
    if (this.state.error) {
      return (
        <div className="p-8 text-center">
          <h2 className="text-lg font-semibold">Something went wrong</h2>
          <pre className="mt-2 text-sm text-red-400">{this.state.error.message}</pre>
        </div>
      )
    }
    return this.props.children
  }
}
