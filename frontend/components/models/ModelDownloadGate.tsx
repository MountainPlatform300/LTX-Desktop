import { Loader2, Download, RefreshCw } from 'lucide-react'
import type { UseIcLoraModelGateResult } from '../../hooks/use-ic-lora-model-gate'

interface ModelDownloadGateProps {
  title: string
  description: string
  gate: UseIcLoraModelGateResult
}

/**
 * Presentational download gate for the IC-LoRA / union-control model bundle.
 * Renders the per-checkpoint progress list, HuggingFace auth + license-access
 * affordances, and the Download / Refresh buttons. State comes from
 * `useIcLoraModelGate`; this component just paints it so the Gen Space LoRA
 * flow can drop it in wherever a union_control LoRA needs preprocessing
 * models that aren't on disk yet.
 */
export function ModelDownloadGate({ title, description, gate }: ModelDownloadGateProps) {
  const {
    checking,
    gateItems,
    error,
    hfAuthStatus,
    hfAuthPolling,
    startHuggingFaceLogin,
    accessMap,
    allAuthorized,
    downloading,
    startDownload,
    refresh,
  } = gate

  return (
    <div className="w-full rounded-xl border border-zinc-700 bg-zinc-800/60 p-4 mb-2">
      <div className="flex items-start gap-3">
        <div className="w-9 h-9 rounded-lg bg-blue-600/20 flex items-center justify-center mt-0.5 shrink-0">
          <Download className="h-4 w-4 text-blue-400" />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold text-white">{title}</h3>
          <p className="text-xs text-zinc-400 mt-1">{description}</p>
        </div>
      </div>

      <div className="mt-4 space-y-3">
        {checking ? (
          <div className="flex items-center gap-2 text-xs text-zinc-300">
            <Loader2 className="h-4 w-4 animate-spin text-blue-400" />
            Checking model availability...
          </div>
        ) : (
          <>
            <div className="space-y-2">
              {gateItems.map((item) => (
                <div key={item.id} className="rounded-lg border border-zinc-700 bg-zinc-900/60 px-3 py-2">
                  <div className="flex items-center justify-between text-[11px] mb-1.5">
                    <span className="text-zinc-300">{item.label}</span>
                    <span className={item.downloaded ? 'text-blue-400' : 'text-zinc-500'}>
                      {item.status}
                    </span>
                  </div>
                  <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className="h-full transition-all duration-300 bg-blue-500"
                      style={{ width: `${item.progress}%` }}
                    />
                  </div>
                  <div className="mt-1 text-[10px] text-zinc-500">{item.progress}%</div>
                </div>
              ))}
            </div>

            {error && (
              <div className="text-[11px] text-red-400">{error}</div>
            )}

            {hfAuthStatus === 'authenticated' && !allAuthorized && Object.keys(accessMap).length > 0 && (
              <div className="space-y-1.5 pt-1 pb-1">
                <div className="text-[11px] text-amber-400">Accept license for these models:</div>
                {Object.entries(accessMap)
                  .filter(([, status]) => status === 'not_authorized')
                  .map(([repoId]) => (
                    <div key={repoId} className="flex items-center justify-between bg-zinc-900 rounded px-2 py-1.5">
                      <span className="text-[10px] text-zinc-400 font-mono">{repoId}</span>
                      <button
                        type="button"
                        onClick={() => window.electronAPI.openHuggingFaceRepo({ repoId })}
                        className="text-[10px] text-indigo-400 hover:text-indigo-300 font-medium"
                      >
                        Request access
                      </button>
                    </div>
                  ))}
              </div>
            )}

            <div className="flex items-center gap-2 pt-1">
              {hfAuthStatus !== 'authenticated' ? (
                <button
                  type="button"
                  onClick={startHuggingFaceLogin}
                  disabled={hfAuthPolling}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {hfAuthPolling ? (
                    <>
                      <Loader2 className="h-3 w-3 animate-spin" />
                      Waiting for sign in...
                    </>
                  ) : (
                    'Sign in with HuggingFace'
                  )}
                </button>
              ) : (
                <button
                  type="button"
                  onClick={startDownload}
                  disabled={downloading || !allAuthorized}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {downloading ? (
                    <>
                      <Loader2 className="h-3 w-3 animate-spin" />
                      Downloading...
                    </>
                  ) : (
                    <>
                      <Download className="h-3 w-3" />
                      {error ? 'Retry Download' : 'Download Models'}
                    </>
                  )}
                </button>
              )}
              <button
                type="button"
                onClick={refresh}
                disabled={checking}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-600 text-zinc-300 hover:text-white hover:border-zinc-500 text-xs transition-colors disabled:opacity-50"
              >
                <RefreshCw className={`h-3 w-3 ${checking ? 'animate-spin' : ''}`} />
                Refresh
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
