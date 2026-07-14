import { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertTriangle, RotateCcw } from 'lucide-react'
import { logger } from '../lib/logger'

type Props = {
  children: ReactNode
  resetKey?: string | number | null
  title?: string
  onReset?: () => void
}

type State = {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    logger.error(
      `UI error boundary: ${error.message}\n${info.componentStack ?? ''}`,
    )
  }

  componentDidUpdate(previous: Props): void {
    if (this.state.error && previous.resetKey !== this.props.resetKey) {
      this.setState({ error: null })
    }
  }

  private reset = (): void => {
    this.setState({ error: null })
    this.props.onReset?.()
  }

  render(): ReactNode {
    if (!this.state.error) return this.props.children

    return (
      <div
        role="alert"
        className="flex flex-1 items-center justify-center bg-background p-6"
      >
        <div className="w-full max-w-md rounded-xl border border-red-500/30 bg-red-500/[0.06] p-5 text-center">
          <AlertTriangle className="mx-auto h-7 w-7 text-red-400" />
          <h2 className="mt-3 text-sm font-semibold text-zinc-100">
            {this.props.title ?? 'This view could not be displayed'}
          </h2>
          <p className="mt-1.5 text-xs leading-relaxed text-zinc-400">
            Your work is still saved. Retry this view; if the problem continues,
            restart LTX Desktop.
          </p>
          <button
            type="button"
            onClick={this.reset}
            className="mt-4 inline-flex items-center gap-1.5 rounded-lg bg-blue-600 px-3 py-2 text-xs font-medium text-white hover:bg-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-400"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Retry view
          </button>
        </div>
      </div>
    )
  }
}
